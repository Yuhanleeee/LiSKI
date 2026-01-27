import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import DeloraConfig, get_peft_model
from tqdm import tqdm


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["instruction"] + item["input"]
        target = item["output"]
        text = prompt + target
        return text


def dynamic_collate_fn(batch, tokenizer, max_length=2048):
    """padding to max length in current batch but do not over than max_length"""
    tokenized = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tokenized["labels"] = labels
    return tokenized


def main():
    # ======= setting =======
    model_id = "models/Qwen2.5-7B-Instruct/"
    train_json_path = "data/ChEBI20_train.json"
    output_dir = "temp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # ======= Tokenizer =======
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ======= Dataset & DataLoader =======
    train_dataset = InstructionDataset(train_json_path, tokenizer, max_length=2048)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: dynamic_collate_fn(x, tokenizer, max_length=2048),
        pin_memory=True,
    )

    # ======= Model + PEFT =======
    peft_config = DeloraConfig(
        r=8,
        delora_lambda=15,
        # target_modules=["all-linear"],
        module_dropout=0.1,
        bias="none",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)

    # ======= Optimizer =======
    lr = 1e-4
    num_epochs = 3
    gradient_accumulation_steps = 8
    warmup_ratio = 0.03

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = len(train_dataloader) * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler()

    # ======= Training =======
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"Epoch {epoch}", ncols=120)

        for step, batch in enumerate(progress):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                avg_loss = total_loss / (step + 1)
                progress.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"[Epoch {epoch}] Avg loss = {total_loss / len(train_dataloader):.4f}")

        # ======= Save =======
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(os.path.join(output_dir, f"epoch-{epoch}"))
        tokenizer.save_pretrained(output_dir)

    print("Save to:", output_dir)


if __name__ == "__main__":
    main()
