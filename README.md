# MDI

PyTorch implementation for **MDI: A Dual-View Deep Learning Framework for Microbe--Drug Association Prediction with Microbial Semantic Representations**.

## Overview

MDI is a dual-view framework for microbe--drug association prediction. The model jointly uses:

- observed microbe--drug association structure,
- homogeneous similarity information from microbe--microbe and drug--drug graphs,
- microbial semantic representations derived from taxonomy-aware text.

The framework learns a homogeneous similarity view and a heterogeneous interaction view, aligns them during training, and fuses the learned representations for downstream prediction.

## Framework

![Framework of MDI](MD.png)

**Figure 1.** Overall framework of **MDI**.

## Usage

The main training and evaluation entry is implemented in `mdi.py`.

### Installation

```bash
git clone https://github.com/yourusername/MDI.git
cd MDI
pip install -r requirements.txt
```

### General usage

Run the main script on a dataset directory:

```bash
python3 mdi.py --dataset_dir ./dataset/MDAD
```

The same calling pattern applies to the other supported datasets:

```bash
python3 mdi.py --dataset_dir ./dataset/aBiofilm
python3 mdi.py --dataset_dir ./dataset/DrugVirus
```

## Project Structure

```text
MDI
│
├── dataset/         # Benchmark datasets
├── mdi.py             # Main training and evaluation script
├── MD.png           # Framework figure
├── README.md        # Project description
└── requirements.txt # Python dependencies
```

## Microbial Semantics

The microbial semantic branch is constructed from taxonomy-aware text. For each microbe, the input text is assembled from the following fields:

- microbe name
- taxonomic rank
- lineage
- synonyms

A generic text template is used to serialize these fields into a single textual description, for example:

```text
Microbe Name: <name>. Rank: <rank>. Lineage: <lineage>. Synonyms: <synonyms>.
```

The resulting text is encoded into a dense vector representation, and pairwise microbial semantic similarity is then computed from these embeddings. In practice, the semantic similarity matrix is used together with the learned microbial semantic embeddings as part of the dual-view framework.

## Datasets

The repository currently supports three datasets:

- `MDAD`
- `aBiofilm`
- `DrugVirus`

Each dataset contains the observed association matrix, microbe/drug similarity matrices, and processed microbial semantic files.

## Citation

If you use this repository in your research, please cite our work.

```bibtex
@article{mdi2026,
  title   = {MDI: A Dual-View Deep Learning Framework for Microbe--Drug Association Prediction with Microbial Semantic Representations},
  author  = {Author information to be added},
  journal = {To be added},
  year    = {2026}
}
```

