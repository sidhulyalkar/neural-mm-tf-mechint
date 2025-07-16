# Multimodal Transformer Demo

## Overview
Demonstrates a transformer-based model that fuses multiple modalities:
- Neural signals (EEG/LFP/spikes)
- Video frames
- Behavioral event sequences
- Task metadata

## Installation
```bash
pip install -r requirements.txt
``` 

## Usage
Generate simulated data and train:
```bash
python train.py --config configs/config.yaml
```