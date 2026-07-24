# AudioTest-DOMAIN ADAPATATION
基于 [AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDomainAdaptation) 项目，添加了 ESC-50 数据集的支持。
> 原始项目：INTERSPEECH 2025 论文 "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification" 的官方代码。

## 环境配置
### 创建 Conda 环境

conda create --name atm-domain-adapt python=3.9
conda activate atm-domain-adapt
pip install -r requirements.txt

额外依赖（用于 HuggingFace 版本）
pip install transformers

数据集准备
ESC-50 数据集下载地址：https://github.com/karolpiczak/ESC-50
将数据放到以下位置：
data/input/esc50/
├── audio/          # 所有 .wav 文件
└── meta/
    └── esc50.csv   # 元数据文件
