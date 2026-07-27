# AudioText-ContextDA with Real Sound Support

This project extends the [AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDomainAdaptation) repository with real-world sound classification experiments.

> Original project: Official release of the INTERSPEECH 2025 paper "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification".
---

## Setup

### Clone repository

```bash
git clone git@github.com:eacevedo1/AudioText-ContextDomainAdaptation.git
cd AudioText-ContextDomainAdaptation
```

### Create environment

```bash
conda create --name atm-domain-adapt python=3.9
conda activate atm-domain-adapt
pip install -r requirements.txt
```

### Extra dependencies (for HuggingFace version)

```bash
pip install transformers
```

---

## Download the pretrained models

### Create the directory

```bash
mkdir -p models/LAION-CLAP
```

### Download the model

```bash
wget -P models/LAION-CLAP https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt
```

---

## Dataset Preparation

### Real Sound Dataset

This project uses a custom real-world sound dataset.

- **Samples**: 28
- **Classes**: 4
- **Format**: WAV

### Place data in the following structure

```text
data/input/real_sound/
├── audio/          # All .wav files
└── meta/
    └── labels.csv  # Metadata file
```

---

## Usage

### 1. Zero-shot classification

```bash
python test_zero_sample.py
```

### 2. Supervised learning with classifier comparison

Supports three classifiers:

- **SVM** (RBFKernel)
- **MLP** (Multi-layer Perceptron)
- **RandomForest** (Random Forest)

```bash
python test_supervised_learning.py
```

### 3. Text-based domain adaptation

Domain adaptation experiment with construction site background across different temperature parameters:

- Temperature: 0.3, 0.5, 0.7

```bash
python test_with_domain_adaptation_text.py
```

---

## Results

### Zero-shot Results

| Metric | Value |
|------|-----|
| Test Samples | 28 |
| Number of Classes | 4 |
| **mAP** | **0.8048** |
| Single-label Accuracy | 32.14% |

### Supervised Learning Results (Classifier Comparison)

| Classifier | mAP | Accuracy |
|--------|-----|--------|
| SVM | 0.9086 | 53.57% |
| RandomForest | 0.9407 | 89.29% |
| **MLP** | **0.9547** | **89.29%** |

### Method Comparison

| Method | Type | mAP | Accuracy |
|--------|------|-----|----------|
| Zero-shot | Zero-shot Learning | 0.8048 | 32.14% |
| SVM | Supervised Learning (RBF Kernel) | 0.9086 | 53.57% |
| RandomForest | Supervised Learning (Ensemble) | 0.9407 | 89.29% |
| **MLP** | **Supervised Learning (Neural Network)** | **0.9547** | **89.29%** |

### Text-based Domain Adaptation Results

Background: **construction_site **

| Temperature | mAP | Single-label Accuracy |
|----------|-----|--------------|
| 0.3 | **0.8080** | 25.00% |
| 0.5 | 0.8053 | 67.86% |
| 0.7 | 0.7852 | 28.57% |

### Key Observations

- **MLP** achieves the best performance with **95.47% mAP**
- **RandomForest** shows strong performance with **94.07% mAP** and **89.29% accuracy**
- **Zero-shot** reaches **80.48% mAP** without any training data
- **Text-based domain adaptation** works best at temperature **0.3** (80.80% mAP)
- Temperature increase from 0.3 to 0.7 causes mAP drop from 80.80% to 78.52%

### Results Location

All results are saved in the `result/` directory:

```
result/
├── zero_sample/                         # Zero-shot results
│   └── zero_sample.csv
├── classifier_comparison/               # Supervised learning results
│   ├── classifier_comparison.csv
│   └── classifier_comparison.png
└── test_with_domain_adaptation_text/    # Domain adaptation results
    ├── temperature_0.3/
    │   └── results.csv
    ├── temperature_0.5/
    │   └── results.csv
    └── temperature_0.7/
        └── results.csv
```


## Citation

If you use this code or ideas from our work, please cite:

> Acevedo, E., Rocamora, M., & Fuentes, M. (2025). Domain Adaptation Method and Modality Gap Impact in Audio-Text Models for Prototypical Sound Classification. In *Interspeech 2025*, pp. 1328-1332.

```bibtex
@inproceedings{acevedo25_interspeech,
  title     = {{Domain Adaptation Method and Modality Gap Impact in Audio-Text Models for Prototypical Sound Classification}},
  author    = {Emiliano Acevedo and Martín Rocamora and Magdalena Fuentes},
  year      = {2025},
  booktitle = {{Interspeech 2025}},
  pages     = {1328--1332},
  doi       = {10.21437/Interspeech.2025-886},
}
```

---

## License

This project is distributed under the same license as the original repository.