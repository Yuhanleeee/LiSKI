# LiSKI - Light and Secure Knowledge Injection

## Quick Start

### Installation
Install LiSKI via pip:
```
Download and Unzip Repository
cd LiSKI/
 
```

### Datasets
Download ChEBI20 dataset from https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data, place it under the `datasets`.
Running `datasets/chebi20_preprocess.py` for dataset preprocessing.
```
# export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download zou-lab/MedCaseReasoning --local-dir datasets
python datasets/medcasereasoning_preproceess.py
```




