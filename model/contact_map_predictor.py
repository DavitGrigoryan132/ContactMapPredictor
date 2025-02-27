import torch
from torch import nn
import torch.nn.functional as F

from .downsampling import DownsamplingBranch
from .upsampling import UpsamplingBranch


class ContactMapPredictor(nn.Module):
    def __init__(
        self, 
        embeddings_size,
        embeddings_processed_size=128,
        embeddings_num_blocks=10,
        similar_structures_number=5,
        structures_channels=3,
        structures_mid_size=64,
        structures_output_size=64,
        structures_num_blocks=3,
        fusion_num_blocks=10,
    ):
        super(ContactMapPredictor, self).__init__()
        self.embeddings_branch = DownsamplingBranch(
            in_channels=embeddings_size * 2,
            num_blocks=embeddings_num_blocks,
            out_channels=embeddings_processed_size
        )
        self.structural_branch = UpsamplingBranch(
            in_channels=similar_structures_number * structures_channels,
            mid_channels=structures_mid_size,
            out_channels=structures_output_size,
            num_blocks=structures_num_blocks
        )
        self.fusion = DownsamplingBranch(
            in_channels=embeddings_processed_size + structures_output_size,
            num_blocks=fusion_num_blocks,
            out_channels=1
        )

    def forward(self, embeddings, structural_tensor):
        embeddings_output = self.embeddings_branch(embeddings)
        structural_output = self.structural_branch(structural_tensor)

        concat_tensor = torch.cat([embeddings_output, structural_output], dim=1)
        out = self.fusion(concat_tensor)
        return out