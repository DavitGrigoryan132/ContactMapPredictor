import os
import re
import torch
import numpy as np
from Bio import SearchIO
from Bio.Blast.Applications import NcbimakeblastdbCommandline, NcbiblastpCommandline


residues_id_map = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "ASN": "N",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}


def get_sequence_from_chain(chain, get_text=False):
    residues = []
    for residue in chain:
        if residue.id[0] == ' ':
            if get_text:
                residues.append(residues_id_map.get(residue.get_resname(), "X"))
            else:
                residues.append(residue)

    if get_text:
        return "".join(residues)
    return residues

def get_sequence_from_model(model, chain_id=None, get_text=False, get_first=False):
    protein_sequences = []
    if get_first:
        protein_sequences.append(get_sequence_from_chain(next(iter(model)), get_text))
    elif chain_id is None:
        for chain in model:
            protein_sequences.append(get_sequence_from_chain(chain, get_text))
    else:
        protein_sequences.append(get_sequence_from_chain(model[chain_id], get_text))

    return protein_sequences


def get_sequence(pdb_filepath, parser, model_id=None, chain_id=None, get_text=False, get_first=False):
    structure = parser.get_structure(os.path.basename(os.path.splitext(pdb_filepath)[0]), pdb_filepath)
    protein_sequences = []
    if get_first:
        protein_sequences += get_sequence_from_model(next(iter(structure)), chain_id, get_text, get_first)
    elif model_id is None:
        for model in structure:
            protein_sequences += get_sequence_from_model(model, chain_id, get_text)
    else:
        protein_sequences += get_sequence_from_model(structure[model_id], chain_id, get_text)

    return protein_sequences

def get_atom_coord(res, atom_name):
    try:
        coord = res[atom_name].get_coord()
        return torch.tensor(coord, dtype=torch.float32)
    except KeyError:
        return None

def compute_inter_residue_information(residues, device):
    """
    Given a list of Bio.PDB.Residue objects,
    compute the inter-residue distance and orientation tensor.

    Returns:
      information_tensor: torch.Tensor of shape (L, L, 3) where for each residue pair (i, j),
      the features are: [distance, theta, phi], computed with respect to the local frame of residue i.
    """
    L = len(residues)
    coords_CA = []
    coords_N  = []
    coords_CB = []

    for res in residues:
        if res.get_resname() not in residues_id_map:
            continue
        ca = get_atom_coord(res, 'CA')
        n  = get_atom_coord(res, 'N')
        cb = get_atom_coord(res, 'CB')
        if ca is None or n is None:
            print(ValueError(f"Residue {res} is missing CA or N."))
            return torch.zeros((len(residues), len(residues), 3), device=device)
        if cb is None:
            # For glycine: pseudo-CB = CA + (CA - N)
            cb = ca + (ca - n)
        coords_CA.append(ca)
        coords_N.append(n)
        coords_CB.append(cb)

    coords_CA = torch.stack(coords_CA, dim=0).to(device)
    coords_N  = torch.stack(coords_N, dim=0).to(device)
    coords_CB = torch.stack(coords_CB, dim=0).to(device)

    diff_CB_CA = coords_CB - coords_CA
    norm_diff = diff_CB_CA.norm(dim=1, keepdim=True) + 1e-8
    e1 = diff_CB_CA / norm_diff

    v = coords_N - coords_CA
    dot = (v * e1).sum(dim=1, keepdim=True)
    proj = v - dot * e1
    norm_proj = proj.norm(dim=1, keepdim=True) + 1e-8
    e2 = proj / norm_proj

    e3 = torch.cross(e1, e2, dim=1)

    diff = coords_CB.unsqueeze(0) - coords_CB.unsqueeze(1)
    d = diff.norm(dim=2, keepdim=True)

    e1_exp = e1.unsqueeze(1)
    e2_exp = e2.unsqueeze(1)
    e3_exp = e3.unsqueeze(1)

    v1 = (diff * e1_exp).sum(dim=2)  # (L, L)
    v2 = (diff * e2_exp).sum(dim=2)  # (L, L)
    v3 = (diff * e3_exp).sum(dim=2)  # (L, L)

    # Compute angles:
    d_nozero = d.squeeze(2) + 1e-8  # (L, L)
    cos_theta = torch.clamp(v1 / d_nozero, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (L, L)
    phi = torch.atan2(v3, v2)       # (L, L)

    # Stack into an tensor of shape (L, L, 3)
    result_tensor = torch.stack([d_nozero, theta, phi], dim=2)
    return result_tensor

def find_similar_sequences(sequence_id, query_sequence, e_value_threshold, db_name, k=10):
    query_file = "query.fasta"

    with open(query_file, "w") as f:
        f.write(f"> {sequence_id} \n")
        f.write(query_sequence)
        f.write('\n\n')

    blast_output = "blast_results.txt"
    blastp_cline = NcbiblastpCommandline(query=query_file, db=db_name, evalue=e_value_threshold,
                                         outfmt=5, out=blast_output)

    stdout, stderr = blastp_cline()

    blast_qresult = SearchIO.read(blast_output, "blast-xml")
    start = 0
    if len(blast_qresult) == 0:
        return []
    if blast_qresult[0].description.strip() == sequence_id:
        k += 1
        start += 1
    similar_chains = []
    for i in range(start, k):
        if i == len(blast_qresult):
            break
        hit = blast_qresult[i]
        hit_description = hit.description.strip()
        regex_match = re.match(r'^([A-Za-z0-9]{4})_(\d)_([A-Za-z])$', hit_description)
        if regex_match is None:
            return None
        structure_id = regex_match.group(1)
        model_id = int(regex_match.group(2))
        chain_id = regex_match.group(3)
        fragments = []
        for hsp in hit:
            alignment = hsp.aln
            fragments.append((alignment[0], alignment[1], hsp.query_start, hsp.hit_start))

        similar_chains.append({
            "sequence_id": sequence_id,
            "sequence": query_sequence,
            "structure_id": structure_id,
            "model_id": model_id,
            "chain_id": chain_id,
            "fragments": fragments
        })

    return similar_chains

def get_structure_from_sequence(sequence_info, dataset_folder, parser, device):
    filename = sequence_info["structure_id"] + ".pdb"
    filepath = os.path.join(dataset_folder, filename)

    structure = parser.get_structure(sequence_info["structure_id"], filepath)
    if "model_id" not in sequence_info:
        model = next(iter(structure))
    else:
        model = structure[sequence_info["model_id"]]
    if "chain_id" not in sequence_info:
        chain = next(iter(model))
    else:
        chain = model[sequence_info["chain_id"]]
    structure_tensor = compute_inter_residue_information(chain, device)
    return structure_tensor

def get_embeddings(model, batch_converter, protein_sequence, device):
    # Convert sequence into ESM2 format
    data = [("protein1", protein_sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])

    # Extract residue embeddings
    residue_embeddings = results["representations"][6][0]

    # First and last embeddings are embeddings of special tokens
    return residue_embeddings[1: -1]

def create_pairwise_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    L, D = embeddings.shape
    emb_i = embeddings.unsqueeze(1)  # (L, 1, D)
    emb_j = embeddings.unsqueeze(0)  # (1, L, D)
    
    pairwise_embeddings = torch.cat([emb_i.expand(L, L, D),
                                     emb_j.expand(L, L, D)], dim=-1)
    return pairwise_embeddings