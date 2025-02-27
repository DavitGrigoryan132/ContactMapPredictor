import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from model import CompoundLoss


class ContactMapLightningModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model: nn.Module = model
        self.learning_rate = learning_rate
        self.loss_fn = CompoundLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=100.0)
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc   = torchmetrics.Accuracy(task="binary")
        self.test_acc  = torchmetrics.Accuracy(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1   = torchmetrics.F1Score(task="binary")
        self.test_f1  = torchmetrics.F1Score(task="binary")   
        self.threshold = 0.75
        
    def forward(self, embeddings_input, structural_input):
        """
        Forward pass:
            embeddings_input: Tensor of shape (B, C₁, L, L)
            structural_input: Tensor of shape (B, C₂, L, L)
        Returns:
            logits: Tensor of shape (B, 1, L, L)
        """
        logits_full = self.model(embeddings_input, structural_input)
        
        B, _, L, _ = logits_full.shape
        
        # Create a mask for upper triangle excluding the diagonal
        mask = torch.triu(torch.ones(B, L, L, dtype=torch.bool, device=logits_full.device), diagonal=1)

        # For each sample in the batch, extract the upper triangle into a vector
        logits_flat = logits_full[:, 0][mask].view(B, -1)
        return logits_flat, mask
    
    def _flatten_target(self, target, mask):
        """
        Given a target contact map of shape (B, L, L) and a mask (L, L),
        flatten each sample's upper triangle (excluding diagonal) to shape (B, num_upper).
        """
        B, L, _ = target.shape
        flattened = target[mask].view(B, -1)
        return flattened

    def training_step(self, batch, batch_idx):
        # Assuming batch size is 1
        pairwise_embeddings, structural_tensor, contact_map = batch[0]
        # Convert pairwise embeddings: (L, L, C₁) -> (1, C₁, L, L)
        embeddings_input = pairwise_embeddings.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # Convert structural tensor: (L, L, C₂) -> (1, C₂, L, L)
        structural_input = structural_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        logits_flat, mask  = self.forward(embeddings_input, structural_input)  

        target = contact_map.unsqueeze(0).to(self.device).float()
        target_flat = self._flatten_target(target, mask)  
        loss = self.loss_fn(logits_flat, target_flat)
        
        B, L, _ = target.shape
        full_logits = torch.zeros((B, 1, L, L), device=self.device)
        full_logits[:, 0][mask] = logits_flat.view(-1)
        preds_flat = full_logits[:, 0][mask]
        
        preds_bin = (torch.sigmoid(preds_flat) >= self.threshold).long()
        target_flat_bin = target_flat.long()
        
        acc = self.train_acc(preds_bin.unsqueeze(0), target_flat_bin)
        f1 = self.train_f1(preds_bin.unsqueeze(0), target_flat_bin)
        
        if (self.global_step + 1) % 100 == 0:
            full_logits = full_logits + full_logits.transpose(2, 3)
            diag_idx = torch.arange(L, device=full_logits.device)
            full_logits[:, 0, diag_idx, diag_idx] = full_logits.max() + 1
            self.visualize(target, full_logits[0], stage="Train")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pairwise_embeddings, structural_tensor, contact_map = batch[0]
        # Convert pairwise embeddings: (L, L, C₁) -> (1, C₁, L, L)
        embeddings_input = pairwise_embeddings.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # Convert structural tensor: (L, L, C₂) -> (1, C₂, L, L)
        structural_input = structural_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        # logits = self.forward(embeddings_input, structural_input)
        logits_flat, mask  = self.forward(embeddings_input, structural_input)  

        target = contact_map.unsqueeze(0).to(self.device).float()
        target_flat = self._flatten_target(target, mask)  
        loss = self.loss_fn(logits_flat, target_flat)
        
        B, L, _ = target.shape
        full_logits = torch.zeros((B, 1, L, L), device=self.device)
        full_logits[:, 0][mask] = logits_flat.view(-1)
        
        preds_flat = full_logits[:, 0][mask]
        
        preds_bin = (torch.sigmoid(preds_flat) >= self.threshold).long()
        target_flat_bin = target_flat.long()
        
        acc = self.val_acc(preds_bin.unsqueeze(0), target_flat_bin)
        f1 = self.val_f1(preds_bin.unsqueeze(0), target_flat_bin)
        
        full_logits = full_logits + full_logits.transpose(2, 3)
        diag_idx = torch.arange(L, device=full_logits.device)
        full_logits[:, 0, diag_idx, diag_idx] = full_logits.max() + 1
        
        self.visualize(target, full_logits[0], stage="Val")
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pairwise_embeddings, structural_tensor, contact_map = batch[0]
        # Convert pairwise embeddings: (L, L, C₁) -> (1, C₁, L, L)
        embeddings_input = pairwise_embeddings.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        # Convert structural tensor: (L, L, C₂) -> (1, C₂, L, L)
        structural_input = structural_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        # logits = self.forward(embeddings_input, structural_input)
        logits_flat, mask  = self.forward(embeddings_input, structural_input)  

        target = contact_map.unsqueeze(0).to(self.device).float()
        target_flat = self._flatten_target(target, mask)  
        loss = self.loss_fn(logits_flat, target_flat)
        
        B, L, _ = target.shape
        full_logits = torch.zeros((B, 1, L, L), device=self.device)
        full_logits[:, 0][mask] = logits_flat.view(-1)
        
        preds_flat = full_logits[:, 0][mask]
        
        preds_bin = (torch.sigmoid(preds_flat) >= self.threshold).long()
        target_flat_bin = target_flat.long()
        print(torch.sigmoid(preds_flat).min(), torch.sigmoid(preds_flat).max())
        print(preds_bin.unique(), target_flat_bin.unique())
        acc = self.val_acc(preds_bin.unsqueeze(0), target_flat_bin)
        f1 = self.val_f1(preds_bin.unsqueeze(0), target_flat_bin)
        
        full_logits = full_logits + full_logits.transpose(2, 3)
        diag_idx = torch.arange(L, device=full_logits.device)
        full_logits[:, 0, diag_idx, diag_idx] = full_logits.max() + 1
        
        self.visualize(target, full_logits[0], stage="Test")
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    
    def visualize(self, target, preds, stage="Test"):
        # Visualization: Create a figure with target and predictions
        target_np = target[0].detach().cpu().numpy()
        preds_np = preds[0].detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(target_np, cmap="gray")
        ax[0].set_title("Ground Truth")
        ax[1].imshow(preds_np, cmap="gray")
        ax[1].set_title("Predictions")
        for a in ax:
            a.axis('off')
        plt.tight_layout()

        if self.logger is not None:
            self.logger.experiment.add_figure(f"{stage}/Predictions_vs_Target", fig, global_step=self.global_step)
        plt.close(fig)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)    
        total_steps = self.trainer.estimated_stepping_batches  # For example, total training steps (you should compute this dynamically)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                anneal_strategy='cos',
                pct_start=0.3,
                div_factor=10,
                final_div_factor=100
            ),
            'interval': 'step',
        }

        return [optimizer], [scheduler]
