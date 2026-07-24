# AudioTest-DOMAIN ADAPATATION

\# AudioText-ContextDA with ESC-50 Support



基于 \[AudioText-ContextDomainAdaptation](https://github.com/eacevedo1/AudioText-ContextDomainAdaptation) 项目，添加了 ESC-50 数据集的支持。



> 原始项目：INTERSPEECH 2025 论文 "Domain Adaptation and Modality Gap in Audio-Text Models for Sound Classification" 的官方代码。



\---



\## Setup



\### Clone repository



```bash

git clone git@github.com:eacevedo1/AudioText-ContextDomainAdaptation.git

cd AudioText-ContextDomainAdaptation



Create environment

conda create --name atm-domain-adapt python=3.9

conda activate atm-domain-adapt

pip install -r requirements.txt

