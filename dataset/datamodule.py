import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import PDBDataset


class PDBDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, test_dir, esm_model, batch_converter, device,
                 database_folder, db_name, e_value_threshold=1e-3, k=10,
                 val_split=0.2, batch_size=1, num_workers=0, max_sequence_length=None):
        """
        Args:
            train_dir (str): Directory with training PDB files.
            test_dir (str): Directory with test PDB files.
            esm_model: Pre-loaded ESM model.
            batch_converter: ESM batch converter function.
            device: torch.device.
            database_folder: folder which contains PDB files from the database
            db_name: BLAST database name
            e_value_threshold: Threshold to find similar sequences
            k: Number of similar sequences to use
            val_split (float): Fraction of training data to use as validation.
            batch_size (int): Batch size.
            num_workers (int): Number of subprocesses for data loading.
            max_sequence_length: Maximum sequence length to use (due to compute limitation)
        """
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.esm_model = esm_model
        self.batch_converter = batch_converter
        self.device = device
        self.database_folder = database_folder
        self.db_name = db_name
        self.e_value_threshold = e_value_threshold
        self.k = k
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length

    def setup(self, stage=None):
        # Create dataset from the training directory
        full_train_dataset = PDBDataset(
            pdb_dir=self.train_dir,
            esm_model=self.esm_model,
            database_folder=self.database_folder,
            db_name=self.db_name,
            e_value_threshold=self.e_value_threshold,
            k=self.k,
            batch_converter=self.batch_converter,
            device=self.device,
            max_sequence_length=self.max_sequence_length
        )

        total_len = len(full_train_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len
        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_len, val_len])

        # Create test dataset from the test directory
        self.test_dataset = PDBDataset(
            pdb_dir=self.test_dir,
            esm_model=self.esm_model,
            database_folder=self.database_folder,
            db_name=self.db_name,
            e_value_threshold=self.e_value_threshold,
            k=self.k,
            batch_converter=self.batch_converter,
            device=self.device,
            max_sequence_length=self.max_sequence_length
        )
        
    def custom_collate(self, batch):
        # Proteins may have different sizes; here we simply return a list.
        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate
        )