
import os
import glob
import torch
from Bio.PDB import PDBParser
from torch.utils.data import Dataset

from .utils import get_sequence, get_structure_from_sequence, get_embeddings, find_similar_sequences, create_pairwise_embeddings

class PDBDataset(Dataset):
    """
    PyTorch Dataset for loading PDB files and computing pairwise embeddings.

    Args:
      pdb_dir (str): Directory containing PDB files.
      esm_model: Pre-loaded ESM model for generating embeddings.
      batch_converter: The ESM batch converter function.
      device: The torch.device to run the model on.
      database_folder: folder which contains PDB files from the database
      db_name: Database name
      e_value_threshold: E value threshold to find similar sequences
      k: Number of similar sequences to use
      max_sequence_length: Max sequence length to use in the model (due to compute limitations)
    """
    def __init__(self, pdb_dir, esm_model, batch_converter, device, database_folder,
                 db_name, e_value_threshold=1e-3, k=10, max_sequence_length=None):
        self.pdb_dir = pdb_dir
        self.database_folder = database_folder
        self.db_name = db_name
        self.e_value_threshold = e_value_threshold
        self.k = k
        self.file_list = glob.glob(os.path.join(pdb_dir, "*.pdb"))
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.device = device
        self.parser = PDBParser()
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def map_structural_info(query_seq, result, query_start, template_start, aligned_query, aligned_template, template_struct_tensor):
        """
        Given:
        - query_seq: the original query sequence (string) of length L.
        - result: Result tensor in which aligned template values should be written
        - query_start: the starting index in the query corresponding to the first character of the aligned query.
        - template_start: the starting index in the template corresponding to the first character of the aligned template.
        - aligned_query, aligned_template: aligned sequences (with '-' gaps).
        - template_struct_tensor: a tensor of shape (T, T, channels) containing structural info from the template.

        Returns:
        - result: a tensor of shape (L, L, channels) where, for positions in the aligned region,
            structural info is copied from template_struct_tensor. Unaligned positions remain zero.
        """
        L = len(query_seq)

        # Build an alignment mapping: for each residue in the ungapped query alignment,
        # map it to the corresponding residue in the template (both indices are offset by query_start and template_start)
        alignment_mapping = {}
        q_idx, t_idx = 0, 0
        original_index = query_start  # index in original sequence

        while q_idx < len(aligned_query) and t_idx < len(aligned_template) and original_index < L:
            # If both positions are non-gap:
            if aligned_query[q_idx] != '-' and aligned_template[t_idx] != '-':
                # Map original query index to template index (offset by template_start)
                alignment_mapping[original_index] = template_start + t_idx
                original_index += 1
                q_idx += 1
                t_idx += 1
            else:
                # If there is a gap in either, increment both indices.
                q_idx += 1
                t_idx += 1

        aligned_query_indices = sorted(alignment_mapping.keys())
        corresponding_template_indices = [alignment_mapping[k] for k in aligned_query_indices]

        q_idx_tensor = torch.tensor(aligned_query_indices, dtype=torch.long)
        t_idx_tensor = torch.tensor(corresponding_template_indices, dtype=torch.long)

        T_dim = template_struct_tensor.shape[0]
        valid_mask = t_idx_tensor < T_dim
        q_idx_tensor = q_idx_tensor[valid_mask]
        t_idx_tensor = t_idx_tensor[valid_mask]

        # Create 2D grids over the aligned indices:
        grid_q_i, grid_q_j = torch.meshgrid(q_idx_tensor, q_idx_tensor, indexing='ij')
        grid_t_i, grid_t_j = torch.meshgrid(t_idx_tensor, t_idx_tensor, indexing='ij')

        result[grid_q_i, grid_q_j, :] = template_struct_tensor[grid_t_i, grid_t_j, :]

        return result

    def __getitem__(self, idx):
        pdb_filepath = self.file_list[idx]
        
        # Read the protein sequence from the PDB file. Using only first chain for simplicity
        protein_sequence = get_sequence(pdb_filepath, self.parser, get_text=True, get_first=True)[0]
        protein_id = os.path.basename(os.path.splitext(pdb_filepath)[0])
        target_contact_map = get_structure_from_sequence({"structure_id": protein_id},
                                                         self.pdb_dir, self.parser, self.device)[..., 0]
        
        # Two residues are considered in contact if the distance between their Cα atoms is below 8 Å (angstroms).
        target_contact_map = (target_contact_map < 8).long()

        # Get only the first self.max_sequence_length sequence due to compute limitations
        if self.max_sequence_length is not None and len(protein_sequence) > self.max_sequence_length:
          protein_sequence = protein_sequence[:self.max_sequence_length]
          target_contact_map = target_contact_map[:self.max_sequence_length, :self.max_sequence_length]

        # Get residue embeddings from ESM model. embeddings should be shape (L, D)
        embeddings = get_embeddings(self.esm_model, self.batch_converter, protein_sequence, self.device)
        pairwise_emb = create_pairwise_embeddings(embeddings)  # shape (L, L, 2*D)

        # Find similar sequences/templates via BLAST search
        similar_sequences = find_similar_sequences(protein_id, protein_sequence, self.e_value_threshold, self.db_name, k=self.k)
        structural_tensor_channels = 3  # We'll use 3 channels (distance, theta, phi) per template
        total_channels = self.k * structural_tensor_channels

        L = len(protein_sequence)
        structural_tensor = torch.zeros((L, L, total_channels), dtype=torch.float32, device=self.device)
        
        # If cannot find similar sequences just return zeros
        # TODO find better solution for this case
        if similar_sequences is None:
            return pairwise_emb, structural_tensor, target_contact_map

        for i, template_info in enumerate(similar_sequences):
            template_structural_tensor = get_structure_from_sequence(template_info, self.database_folder, self.parser, self.device)

            for aligned_query, aligned_template, query_start, template_start in template_info["fragments"]:
                structural_tensor[..., i * structural_tensor_channels: (i + 1) * structural_tensor_channels] = \
                    self.map_structural_info(template_info["sequence"], structural_tensor[..., i * structural_tensor_channels: (i + 1) * structural_tensor_channels],
                                             query_start, template_start, aligned_query, aligned_template, template_structural_tensor[:, :, :structural_tensor_channels])

        tensor_reshaped = structural_tensor.view(-1, total_channels)
        mean = tensor_reshaped.mean(dim=0, keepdim=True)
        std = tensor_reshaped.std(dim=0, keepdim=True) + 1e-8
        normalized = (tensor_reshaped - mean) / std
        normalized = normalized.view(L, L, total_channels)

        return pairwise_emb, normalized, target_contact_map