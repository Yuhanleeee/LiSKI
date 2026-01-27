# LiSKI - Light and Secure Knowledge Injection

## Quick Start

### Installation
Install LiSKI via pip:
```
Download and Unzip Repository
cd LiSKI/
conda create -n LiSKI python=3.10
conda activate LiSKI
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Datasets
**ChEBI20**:  
Download ChEBI20 dataset from https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data, place it under the `datasets`.
Running `datasets/chebi20_preprocess.py` for dataset preprocessing.

**MedCaseReasoning**:

```
# export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download zou-lab/MedCaseReasoning --local-dir datasets
python datasets/medcasereasoning_preproceess.py
```




