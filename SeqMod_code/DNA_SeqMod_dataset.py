import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BindingDataset(Dataset):
    def __init__(self, esm_dir, data_list, data_df):
        self.esm_dir = esm_dir
        self.data_list = data_list
        self.data_df = data_df.set_index("ID")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        protein_id = self.data_list[index]

        feature_path = os.path.join(self.esm_dir, protein_id + ".pkl")

        with open(feature_path, "rb") as f:
            data = pickle.load(f)

        seq_embedding = torch.tensor(
            data["seq_embeding"],
            dtype=torch.float
        )

        dna_annotation = self.data_df.loc[protein_id, "DNA"]
        dna_label = np.array([[int(x)] for x in dna_annotation])

        return seq_embedding, dna_label


def fasta_to_dataframe(file_path):
    ids = []
    sequences = []
    annotations = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for i in range(0, len(lines) - 2, 3):
        ids.append(lines[i].strip()[1:])
        sequences.append(lines[i + 1].strip())
        annotations.append(lines[i + 2].strip())

    df = pd.DataFrame({
        "ID": ids,
        "Sequence": sequences,
        "IDR": annotations
    })

    for col in ["PB", "NB", "LB", "IB", "SB", "DNA"]:
        df[col] = df["IDR"]

    return df


def create_dataloader(fasta_path, esm_dir,
                      batch_size=1,
                      shuffle=False):

    df = fasta_to_dataframe(fasta_path)
    data_list = df["ID"].tolist()

    dataset = BindingDataset(
        esm_dir=esm_dir,
        data_list=data_list,
        data_df=df
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def check_dataloader(dataloader):
    for i, data in enumerate(dataloader):
        if i >= 2:
            break

        feature, dna_label = data

        print("Feature size:", feature.size())
        print("DNA label size:", dna_label.size())


if __name__ == "__main__":

    esm_feature_dir = "YOUR_ESM_FEATURE_DIR"

    train_dataloader = create_dataloader(
        fasta_path="TRAIN_FASTA",
        esm_dir=esm_feature_dir
    )

    valid_dataloader = create_dataloader(
        fasta_path="VALID_FASTA",
        esm_dir=esm_feature_dir
    )

    test129_dataloader = create_dataloader(
        fasta_path="TEST129_FASTA",
        esm_dir=esm_feature_dir
    )

    dataloaders = [
        train_dataloader,
        valid_dataloader,
        test129_dataloader
    ]

    for i, loader in enumerate(dataloaders):
        print(f"Checking DataLoader {i + 1}:")
        check_dataloader(loader)
        print(len(loader))