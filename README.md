# AudioText-ContextDA: Audio-Text Domain Adaptation Experiments

This project is built upon the [AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDA) repository and contains two independent audio classification experiments.

> Original project: Official release of the INTERSPEECH 2025 paper "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification".

---

## Project Overview

| Subproject | Dataset | Methods | Special Settings | Metric |
|------------|---------|---------|------------------|--------|
| [ESC-50 Experiment](./AudioText-ContextDA-main-ESC50) | ESC-50 (2,000 samples, 50 classes) | Zero-shot, KNN, SVM | None | Accuracy |
| [Real Sound Experiment](./AudioText-ContextDA-main-REAL) | Real-world sounds (28 samples, 4 classes) | Zero-shot, Supervised Learning (SVM/MLP/RF), Text-based Domain Adaptation | Temperature (construction site) 0.3/0.5/0.7 | mAP |

---

## Environment Setup

Both subprojects share the same environment:

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

## Directory Structure

```
AudioTest-real/
├── README.md                                    # Root README (this file)
├── AudioText-ContextDA-main-ESC50/              # ESC-50 Experiment
│   ├── README.md
│   ├── process_esc50_hf.py
│   ├── test_knn.py
│   ├── test_zeroshot.py
│   ├── test_svm.py
│   ├── class_labels_esc50.txt
│   └── results/
└── AudioText-ContextDA-main-REAL/               # Real Sound Experiment
    ├── README.md
    ├── test_zero_sample.py
    ├── test_supervised_learning.py
    ├── test_with_domain_adaptation_text.py
    └── result/
        ├── zero_sample/
        ├── classifier_comparison/
        └── test_with_domain_adaptation_text/
            ├── temperature_0.3/
            ├── temperature_0.5/
            └── temperature_0.7/
```

---

## Results Summary

### ESC-50 Experiment Results (Accuracy)

| Fold | Zero-shot | KNN | SVM |
|------|-----------|-----|-----|
| Fold 1 | 84.75% | 89.50% | 90.00% |
| Fold 2 | 75.50% | 86.25% | 87.50% |
| Fold 3 | 79.75% | 88.25% | 88.75% |
| Fold 4 | 79.75% | 84.75% | 90.50% |
| Fold 5 | 86.50% | 90.75% | 91.75% |
| **Mean** | **81.25%** | **87.90%** | **89.70%** |
| **Std** | ±3.91% | ±2.17% | ±1.60% |

### Real Sound Experiment Results

#### Zero-shot 

| Metric | Value |
|------|-----|
| Test Samples | 28 |
| Number of Classes | 4 |
| **mAP** | **0.8048** |
| Single-label Accuracy | 32.14% |

#### Supervised Learning (Classifier Comparison)

| Classifier | mAP | Accuracy |
|--------|-----|--------|
| SVM | 0.9086 | 53.57% |
| RandomForest | 0.9407 | 89.29% |
| **MLP** | **0.9547** | **89.29%** |

> **Best Classifier: MLP (mAP = 0.9547)**

#### Text-based Domain Adaptation (Construction Site Background)

| Temperature | mAP | Single-label Accuracy |
|----------|-----|--------------|
| 0.3 | **0.8080** | 25.00% |
| 0.5 | 0.8053 | 67.86% |
| 0.7 | 0.7852 | 28.57% |

> **Best Temperature: 0.3(mAP = 0.8080)**

---

## Key Findings

1. **Supervised learning achieves the best performance**:MLP reaches 95.47% mAP,significantly outperforming Zero-shot (80.48%)
2. **RandomForest performs strongly**:  94.07% mAP with 89.29% accuracy, comparable to MLP
3. **Text-based domain adaptation**: Optimal performance at temperature 0.3 (80.80% mAP)
4. **Temperature impact**: mAP drops from 80.80% to 78.52% as temperature increases from 0.3 to 0.7

---

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
