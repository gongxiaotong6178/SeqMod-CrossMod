import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BindingDataset(Dataset):
    def __init__(self, saprot_dir, esm2_dir, data_list, data_df):
        self.saprot_dir = saprot_dir
        self.esm2_dir = esm2_dir
        self.data_list = data_list
        self.data_df = data_df.set_index("ID")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        protein_id = self.data_list[index]

        saprot_path = os.path.join(self.saprot_dir, protein_id + ".pkl")
        with open(saprot_path, "rb") as f:
            saprot_data = pickle.load(f)

        saprot_feat = torch.as_tensor(
            saprot_data["seq_embeding"],
            dtype=torch.float32
        )

        esm2_path = os.path.join(self.esm2_dir, protein_id + ".pkl")
        with open(esm2_path, "rb") as f:
            esm2_data = pickle.load(f)

        esm2_feat = torch.as_tensor(
            esm2_data["seq_embeding"],
            dtype=torch.float32
        )

        dna_annotation = self.data_df.loc[protein_id, "DNA"]
        dna_label = np.array(
            [[int(x)] for x in dna_annotation],
            dtype=np.int64
        )

        dna_label = torch.as_tensor(
            dna_label,
            dtype=torch.float32
        )

        return esm2_feat, saprot_feat, dna_label


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

    df["DNA"] = df["IDR"]

    return df


def create_dataloader(
    fasta_path,
    saprot_dir,
    esm2_dir,
    batch_size=1,
    shuffle=False
):
    df = fasta_to_dataframe(fasta_path)
    data_list = df["ID"].tolist()

    dataset = BindingDataset(
        saprot_dir=saprot_dir,
        esm2_dir=esm2_dir,
        data_list=data_list,
        data_df=df
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataloader


def check_dataloader(dataloader):
    for i, data in enumerate(dataloader):
        if i >= 2:
            break

        esm_feat, saprot_feat, dna_label = data

        print("ESM feature size:", esm_feat.size())
        print("SaProt feature size:", saprot_feat.size())
        print("DNA label size:", dna_label.size())


if __name__ == "__main__":

    saprot_feature_dir = "YOUR_SAPROT_FEATURE_DIR"
    esm2_feature_dir = "YOUR_ESM2_FEATURE_DIR"

    train_dataloader = create_dataloader(
        fasta_path="TRAIN_FASTA",
        saprot_dir=saprot_feature_dir,
        esm2_dir=esm2_feature_dir
    )

    valid_dataloader = create_dataloader(
        fasta_path="VALID_FASTA",
        saprot_dir=saprot_feature_dir,
        esm2_dir=esm2_feature_dir
    )

    test129_dataloader = create_dataloader(
        fasta_path="TEST129_FASTA",
        saprot_dir=saprot_feature_dir,
        esm2_dir=esm2_feature_dir
    )

    dataloaders = [
        train_dataloader,
        valid_dataloader,
        test129_dataloader
    ]

    for i, loader in enumerate(dataloaders):
        print(f"Checking DataLoader {i + 1}:")
        check_dataloader(loader)
        print("Total samples:", len(loader))
        print()