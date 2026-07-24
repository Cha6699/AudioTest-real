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

```



\### Create environment



```bash

conda create --name atm-domain-adapt python=3.9

conda activate atm-domain-adapt

pip install -r requirements.txt

```



\### Extra dependencies (for HuggingFace version)



```bash

pip install transformers

```



\---



\## Download the pretrained models



\### Create the directory



```bash

mkdir -p models/LAION-CLAP

```



\### Download the model



```bash

wget -P models/LAION-CLAP https://huggingface.co/lukewys/laion\_clap/resolve/main/630k-audioset-fusion-best.pt

```



\---



\## Dataset Preparation



\### Download ESC-50 dataset



Download from: https://github.com/karolpiczak/ESC-50



\### Place data in the following structure



```

data/input/esc50/

├── audio/          # All .wav files

└── meta/

&#x20;   └── esc50.csv   # Metadata file

```



\---



\## Usage



\### 1. Extract ESC-50 audio embeddings



```bash

python process\_esc50\_hf.py

```



> The script uses HuggingFace `laion/clap-htsat-fused` model, which will be downloaded automatically.



\### 2. Run classification tests



\#### KNN classification



```bash

python test\_knn.py

```



\#### Zero-shot classification



```bash

python test\_zeroshot.py

```



\#### SVM supervised classification



```bash

python test\_svm.py

```



\### 3. Test a single audio file



```bash

python predict\_single\_audio.py

```



\---



\## Results



| Method | Mean Accuracy |

|--------|---------------|

| KNN (Cosine Distance) | 87.90% |

| Zero-shot | TBD |

| SVM (RBF) | TBD |



> Results are saved in `results/` directory



\---



\## Citation



If you use this code or ideas from our work, please cite:



> @inproceedings{acevedo25\_interspeech,

>   title     = {{Domain Adaptation Method and Modality Gap Impact in Audio-Text Models for Prototypical Sound Classification}},

>   author    = {Emiliano Acevedo and Martín Rocamora and Magdalena Fuentes},

>   year      = {2025},

>   booktitle = {{Interspeech 2025}},

>   pages     = {1328--1332},

>   doi       = {10.21437/Interspeech.2025-886},

> }

\---



\## License



This project is distributed under the same license as the original repository.

