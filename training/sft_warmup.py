import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from dataclasses import dataclass
from .configs import SFTWarmupConfig

def _collate_fn(batch, pad_id: int, eos_id: int):
    """Pad a training batch to uniform length with EOS appended."""
    padded = []
    for b in batch:
        padded.append({
            "input_ids": torch.cat([b["input_ids"], torch.tensor([eos_id])]),
            "labels": torch.cat([b["labels"], torch.tensor([eos_id])]),
            "attention_mask": torch.cat([b["attention_mask"], torch.ones(1)]),
        })

    max_len = max(b["input_ids"].shape[0] for b in padded)

    input_ids = []
    labels = []
    attention_mask = []

    for b in padded:
        pad_len = max_len - b["input_ids"].shape[0]
        input_ids.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([b["attention_mask"], torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def sft_warmup(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    config: SFTWarmupConfig = SFTWarmupConfig(),
):
    """
    Short SFT warmup to teach a base model the <think> token structure
    before GRPO training. Only trains embeddings and lm_head.
    """
    print("=" * 50)
    print("  SFT WARMUP")
    print("=" * 50)

    # ── Freeze everything, unfreeze embeddings + head ──
    for param in model.parameters():
        param.requires_grad = False

    model.get_input_embeddings().weight.requires_grad = True
    model.lm_head.weight.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {n_trainable:,} / {n_total:,} params ({100*n_trainable/n_total:.2f}%)")
    print(f"  Dataset:   {len(dataset)} examples")
    print(f"  Epochs:    {config.epochs}, Batch size: {config.batch_size}")
    print("=" * 50)

    optimizer = AdamW(trainable_params, lr=config.lr)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate_fn(batch, pad_id, eos_id),
    )

    # ── Training loop ──
    model.train()
    global_step = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                print(f"  Step {global_step:>4d} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{config.epochs} | Avg Loss: {avg_loss:.4f}")

    # ── Restore model state ──
    for param in model.parameters():
        param.requires_grad = True
    model.eval()

    # Clean up optimizer to free VRAM
    del optimizer
    torch.cuda.empty_cache()

    print("=" * 50)
    print(f"  SFT warmup complete. {global_step} steps.")
    print("=" * 50)