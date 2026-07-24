# AudioText-ContextDA with ESC-50 Support

基于 [AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDomainAdaptation) 项目，添加了 ESC-50 数据集的支持。

> 原始项目：INTERSPEECH 2025 论文 "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification" 的官方代码。

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

### Download ESC-50 dataset

Download from: https://github.com/karolpiczak/ESC-50

### Place data in the following structure

data/input/esc50/
├── audio/          # All .wav files
└── meta/
    └── esc50.csv   # Metadata file

---

## Usage

### 1. Extract ESC-50 audio embeddings

```bash
python process_esc50_hf.py
```

> The script uses HuggingFace `laion/clap-htsat-fused` model, which will be downloaded automatically.

### 2. Run classification tests

#### KNN classification

```bash
python test_knn.py
```

#### Zero-shot classification

```bash
python test_zeroshot.py
```

#### SVM supervised classification

```bash
python test_svm.py
```

### 3. Test a single audio file

```bash
python predict_single_audio.py
```

---

## Results

### ESC-50 Classification Results (5-fold Cross Validation)

| Fold | Zero-shot | KNN | SVM |
|------|-----------|-----|-----|
| Fold 1 | 84.75% | 89.50% | 90.00% |
| Fold 2 | 75.50% | 86.25% | 87.50% |
| Fold 3 | 79.75% | 88.25% | 88.75% |
| Fold 4 | 79.75% | 84.75% | 90.50% |
| Fold 5 | 86.50% | 90.75% | 91.75% |
| **Mean** | **81.25%** | **87.90%** | **89.70%** |
| **Std** | ±3.91% | ±2.17% | ±1.60% |

### Method Comparison

| Method | Type | Mean Accuracy | Std |
|--------|------|---------------|-----|
| Zero-shot | 零样本学习 | 81.25% | ±3.91% |
| KNN | 监督学习 (余弦距离) | 87.90% | ±2.17% |
| SVM | 监督学习 (RBF核) | **89.70%** | **±1.60%** |

### Key Observations

- **SVM** achieves the best performance with **89.70%** accuracy
- **Zero-shot** reaches **81.25%** without any training data
- **SVM** shows the most stable performance across all folds (±1.60%)

### Results Location

All results are saved in the `results/` directory:
- `knn_YYYYMMDD_HHMMSS/`
- `zeroshot_YYYYMMDD_HHMMSS/`
- `svm_YYYYMMDD_HHMMSS/`

---

## Project Structure

```
AudioText-ContextDA/
├── scripts/                    # Original project scripts
│   ├── download_dataset.py
│   ├── extract_embeddings.py
│   ├── inference_classification.py
│   ├── sound_classification.py
│   └── soundscape_augmentations.py
├── src/                        # Original project source
│   ├── domain_adaptation_utils.py
│   ├── get_datasets.py
│   ├── get_embedding.py
│   ├── get_models.py
│   └── ...
├── process_esc50_hf.py         # ESC-50 embedding extraction
├── predict_single_audio.py     # Single audio inference
├── test_knn.py                 # KNN classification
├── test_zeroshot.py            # Zero-shot classification
├── test_svm.py                 # SVM classification
├── class_labels_esc50.txt      # ESC-50 class labels
├── results/                    # Test results
├── data/                       # Dataset directory
│   └── input/
│       └── esc50/
│           ├── audio/          # ESC-50 audio files
│           └── meta/
│               └── esc50.csv
├── models/                     # Pretrained models
│   └── LAION-CLAP/
│       └── 630k-audioset-fusion-best.pt
└── README.md
```

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
