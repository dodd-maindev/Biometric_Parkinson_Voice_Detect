# Voice-Based Parkinson's Disease Screening Benchmark

This repository implements a Clean Architecture benchmark for screening Parkinson's Disease (PD) from speech recordings using the MDVR-KCL and NeuroVoz datasets.

## Architecture & Principles
- **Clean Architecture & Strict OOP**: Complete decoupling between Domain, Infrastructure, and Application layers.
- **Leakage-Free Validation**: Strict Leave-One-Subject-Out Cross-Validation (LOSOCV).
- **Benchmarking**: Replicates Hossain et al. (Neurol. Int. 2026) and probes Self-Supervised Speech Models (Wav2Vec2, WavLM).

## Repository Structure
```
src/
├── domain/            # Entities, Enums, and Abstract Interfaces
├── infrastructure/    # Audio segmenters, Praat/GTCC/MFCC/SSL extractors, ML models
└── application/       # Pipelines (EXP-0, EXP-1, EXP-2), LOSOCV strategy, Metric services
```

## Running on Google Colab Pro+

### 1. Install Dependencies
```bash
!pip install -r requirements.txt
!apt-get install -y ffmpeg
```

### 2. Download MDVR-KCL Dataset
```bash
!mkdir -p data/raw/mdvr_kcl
# Download directly from Zenodo (Record 2867216)
!wget -O data/raw/mdvr_kcl.zip https://zenodo.org/records/2867216/files/MDVR_KCL.zip?download=1
!unzip -q data/raw/mdvr_kcl.zip -d data/raw/mdvr_kcl/
```

### 3. Run Experiments

#### EXP-0: Baseline Replication (Acoustic + GTCC + MFCC + SVM)
```bash
!python run_experiments.py --experiment baseline --task READ_TEXT
```

#### EXP-1: Frozen SSL Embedding (Wav2Vec2-base + LOSOCV)
```bash
!python run_experiments.py --experiment ssl_frozen --model_name facebook/wav2vec2-base --task READ_TEXT
```

#### EXP-2: Layer-wise Probing (e.g., Layer 6)
```bash
!python run_experiments.py --experiment ssl_probe --model_name facebook/wav2vec2-base --layer_index 6 --task READ_TEXT
```
