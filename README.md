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
Download the ChEBI20 dataset from https://github.com/blender-nlp/MolT5/tree/main/ChEBI-20_data, and place it under the `datasets`.
Run `datasets/chebi20_preprocess.py` for dataset preprocessing.

**MedCaseReasoning**:

```
# export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download --repo-type dataset --resume-download zou-lab/MedCaseReasoning --local-dir datasets
python datasets/medcasereasoning_preproceess.py
```

### Knowledge Injection and Calibration

- **Step 1.** Fine-tuning SLM with a specific dataset:
  ```
  cd LLaMA-Factory/
  LLamafactory-cli train examples/train_full/qwen25_full_sft.yaml model_name_or_path=Qwen2.5-0.5B-Instruct/ dataset=ChEBI20 
  ```
- **Step 2.** Fine-tuning SLM with a specific calibration dataset:
  ```
  cd LLaMA-Factory/
  LLamafactory-cli train examples/train_full/qwen25_full_sft.yaml model_name_or_path=Qwen2.5-0.5B-Instruct/ dataset=ChEBI20_Calibration
  ```
- **Step 3.** Fine-tuning LLM with a specific calibration dataset:
  Edit `LlamaFactory/src/llamafactory/train/sft/workflow.py` by inserting the provided code to freeze specific layers (no knowledge injection layers) of the LLM.
  Then, launch training with the CLI: 
  ```
  LLamafactory-cli train examples/train_full/qwen25_full_sft.yaml model_name_or_path=Qwen2.5-3B-Instruct/ dataset=ChEBI20_Calibration
  ```
- **Step 4.** Obtaining knowledge transfer matrices and injecting knowledgeable parameters:
  ```
  python knowledge_injection.py
  ```
- **Step 5.** Knowledge calibration via the script:
  ```
  python knowledge_calibration.py
  ``` 

### TopK-Layer Selection
- **Step 1.** Reload `transformers/trainer.py` with `grad_score/trainer.py`.
- **Step 2.** Launch training with the CLI:
  ```
  cd LLaMA-Factory/
  LLamafactory-cli train examples/train_full/qwen25_full_sft.yaml model_name_or_path=Qwen2.5-3B-Instruct/ dataset=ChEBI20_Calibration num_train_epochs=1.0
  ```
- **Step 3.** Run the script below for TopK layer selection:
  ```
  python grad_score/grad_score.py
  ```

## Citation
```

```


