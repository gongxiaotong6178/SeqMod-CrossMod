import os
import pickle
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class BindingDataset(Dataset):
    def __init__(self, esm_dir, data_list, data_df):
        self.data_list = data_list
        self.data_df = data_df.set_index("ID")
        self.esm_dir = esm_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        protein_id = self.data_list[index]

        embedding_path = os.path.join(self.esm_dir, protein_id + ".pkl")
        with open(embedding_path, "rb") as f:
            data = pickle.load(f)

        seq_embedding = torch.tensor(
            data["seq_embeding"],
            dtype=torch.float
        )

        rna_annotation = self.data_df.loc[protein_id, "RNA"]
        rna_label = np.array([[int(x)] for x in rna_annotation])

        return seq_embedding, rna_label


def fasta_to_dataframe(file_name):
    ids = []
    sequences = []
    annotations = []

    with open(file_name, "r") as f:
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

    for column in ["PB", "NB", "LB", "IB", "SB", "RNA"]:
        df[column] = df["IDR"]

    return df


ESM2_FEATURE_PATH = "./embedding1280_all/"


train_df = fasta_to_dataframe(
    "./RNA_train.fasta"
)
train_dataset = BindingDataset(
    ESM2_FEATURE_PATH,
    train_df["ID"].tolist(),
    train_df
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False
)


valid_df = fasta_to_dataframe(
    "./RNA_valid.fasta"
)
valid_dataset = BindingDataset(
    ESM2_FEATURE_PATH,
    valid_df["ID"].tolist(),
    valid_df
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False
)


test_df = fasta_to_dataframe(
    "./RNA_Test117.fasta"
)
test_dataset = BindingDataset(
    ESM2_FEATURE_PATH,
    test_df["ID"].tolist(),
    test_df
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


def check_dataloader(dataloader):
    for i, data in enumerate(dataloader):
        if i < 2:
            feature, rna_label = data
            print("Feature size:", feature.size())
            print("RNA label size:", rna_label.size())


if __name__ == "__main__":
    dataloaders = [
        train_dataloader,
        valid_dataloader,
        test_dataloader
    ]

    for i, dataloader in enumerate(dataloaders):
        print(f"Checking DataLoader {i + 1}:")
        check_dataloader(dataloader)
        print(len(dataloader))