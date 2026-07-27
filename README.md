# AudioText-ContextDA: 音频文本领域适配实验

本项目基于 [AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDomainAdaptation) 项目，包含两组独立的音频分类实验。

> 原始项目：INTERSPEECH 2025 论文 "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification" 的官方代码。

---

## 项目概览

| 子项目 | 数据集 | 测试方法 | 特殊设置 | 评估指标 |
|--------|--------|----------|----------|----------|
| [ESC-50 实验](./AudioText-ContextDA-main-ESC50) | ESC-50 (2000个样本, 50类) | Zero-shot, KNN, SVM | 无 | Accuracy |
| [真实声音实验](./AudioText-ContextDA-main-REAL) | 真实录制声音 (28个样本, 4类) | Zero-shot, 监督学习 (SVM/MLP/RF), 文本领域适配 | 温度背景 (工地) 0.3/0.5/0.7 | mAP |

---

## 环境配置

两个子项目共用同一个环境：

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

## 目录结构

```
AudioTest-real/
├── README.md                                    # 总 README (本文件)
├── AudioText-ContextDA-main-ESC50/              # ESC-50 实验
│   ├── README.md
│   ├── process_esc50_hf.py
│   ├── test_knn.py
│   ├── test_zeroshot.py
│   ├── test_svm.py
│   ├── class_labels_esc50.txt
│   └── results/
└── AudioText-ContextDA-main-REAL/               # 真实声音实验
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

## 结果汇总

### ESC-50 实验结果 (Accuracy)

| Fold | Zero-shot | KNN | SVM |
|------|-----------|-----|-----|
| Fold 1 | 84.75% | 89.50% | 90.00% |
| Fold 2 | 75.50% | 86.25% | 87.50% |
| Fold 3 | 79.75% | 88.25% | 88.75% |
| Fold 4 | 79.75% | 84.75% | 90.50% |
| Fold 5 | 86.50% | 90.75% | 91.75% |
| **Mean** | **81.25%** | **87.90%** | **89.70%** |
| **Std** | ±3.91% | ±2.17% | ±1.60% |

### 真实声音实验结果

#### Zero-shot (零样本)

| 指标 | 值 |
|------|-----|
| 测试样本数 | 28 |
| 类别数 | 4 |
| **mAP** | **0.8048** |
| 单标签准确率 | 32.14% |

#### 监督学习 (分类器对比)

| 分类器 | mAP | 准确率 |
|--------|-----|--------|
| SVM | 0.9086 | 53.57% |
| RandomForest | 0.9407 | 89.29% |
| **MLP** | **0.9547** | **89.29%** |

> **最佳分类器：MLP (mAP = 0.9547)**

#### 文本领域适配 (工地背景)

| 温度参数 | mAP | 单标签准确率 |
|----------|-----|--------------|
| 0.3 | **0.8080** | 25.00% |
| 0.5 | 0.8053 | 67.86% |
| 0.7 | 0.7852 | 28.57% |

> **最佳温度：0.3 (mAP = 0.8080)**

---

## 关键发现

1. **监督学习效果最好**: MLP 达到 **95.47% mAP**，远超 Zero-shot (80.48%)
2. **RandomForest 表现优异**: 89.29% 准确率，接近 MLP
3. **文本领域适配**: 温度参数在 0.3 时效果最佳 (80.80% mAP)
4. **温度影响**: 温度从 0.3 升至 0.7，mAP 从 80.80% 降至 78.52%

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