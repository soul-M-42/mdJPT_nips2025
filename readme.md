# mdJPT: Multi-dataset Joint Pretrain Transformer for Emotion Decoder from EEG

## Project Overview
This project implements the code from the paper [mdJPT: Multi-dataset Joint Pretrain Transformer for Emotion Decoder from EEG](url). It is designed for joint training of emotion classification models across multiple EEG datasets.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Run the Pipeline
```bash
bash run_pipeline.sh
```

## Introduction

- **train_multi.py**: Pretrain feature extractor model.  
- **ext_fea.py**: Extract features using the pretrained model.  
- **train_mlp_full.py**: Perform cross-subject emotion validation on extracted features by training an MLP.

## Configuration

All training and validation parameters are defined in the configuration file `cfgs_multi/config_multi.yaml`.  
If you want to modify hyperparameters, model settings, or dataset options for your experiments, please edit this YAML file accordingly.  


