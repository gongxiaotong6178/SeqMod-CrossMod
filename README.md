# SeqMod and CrossMod for Protein–Nucleic Acid Binding Site Prediction

## Overview

This repository contains the implementation of two deep learning frameworks for protein–nucleic acid binding site prediction:

* **SeqMod**: a sequence-based prediction framework using protein language model embeddings.
* **CrossMod**: a multimodal prediction framework integrating sequence and structure-aware protein representations.

Both frameworks support **DNA-binding** and **RNA-binding** residue prediction tasks.

---

## Repository Structure

├── SeqMod/
│   ├── DNA_dataset.py
│   ├── DNA_model.py
│   ├── RNA_dataset.py
│   └── RNA_model.py
│
├── CrossMod/
│   ├── DNA_dataset.py
│   ├── DNA_model.py
│   ├── RNA_dataset.py
│   └── RNA_model.py
│
└── README.md

### SeqMod

SeqMod is a sequence-driven framework based on protein language model embeddings.

* DNA_SeqMod_dataset.py

  * Data loading for the DNA-binding prediction task.
  * Loads ESM2 embeddings and residue-level labels.

* DNA_SeqMod_model.py

  * SeqMod model implementation for DNA-binding site prediction.

* RNA_SeqMod_dataset.py

  * Data loading for the RNA-binding prediction task.
  * Loads ESM2 embeddings and residue-level labels.

* RNA_SeqMod_model.py

  * SeqMod model implementation for RNA-binding site prediction.

---

### CrossMod

CrossMod is a multimodal framework that combines sequence and structure-aware protein representations.

* DNA_CrossMod_dataset.py

  * Data loading for DNA-binding prediction.
  * Loads both ESM2 and SaProt embeddings.

* DNA_CrossMod_model.py

  * CrossMod model implementation for DNA-binding site prediction.

* RNA_CrossMod_dataset.py

  * Data loading for RNA-binding prediction.
  * Loads both ESM2 and SaProt embeddings.

* RNA_CrossMod_model.py

  * CrossMod model implementation for RNA-binding site prediction.

---

## Input Features

### SeqMod

* ESM2 protein embeddings.
* Residue-level binding annotations.

### CrossMod

* ESM2 protein embeddings.
* SaProt structure-aware embeddings.
* Residue-level binding annotations.

Embedding files should be stored in `.pkl` format, where each file contains:

{
    "seq_embeding": embedding_matrix
}


---

## Data Preparation

The dataset loaders require:

* Protein IDs;
* Protein sequences;
* Residue-level binding annotations.

FASTA files should follow the format:

```
>Protein_ID
SEQUENCE
ANNOTATION
```

where the annotation string consists of residue-level binary labels.

---

## Usage

Before running the scripts, specify the locations of:

* embedding directories;
* FASTA files.

The dataset loaders are designed to be easily adapted to custom datasets by modifying the input paths.

---

## Requirements

* Python 3.8
* PyTorch
* NumPy
* Pandas

---

## Citation

If you use this code in your research, please cite our corresponding publication.

---

## Contact

For questions regarding the implementation, please contact the corresponding author.
