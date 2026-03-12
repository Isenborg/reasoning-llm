from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEFAULT_MODEL_DIR = Path("/nobackup") / Path.home().name / "models"
# resolves to: /nobackup/liuid123/models


def load_model(model_name: str, model_dir: Path = DEFAULT_MODEL_DIR, **kwargs):
    """
    Load model. Uses local cache if available,
    otherwise downloads from HuggingFace and saves locally.
    """
    if not model_dir.parent.exists():
        raise FileNotFoundError(
            f"Directory '{model_dir.parent}' does not exist.\n"
            f"You have to create it before continuing!"
        )

    local_path = model_dir / model_name
    config_file = local_path / "config.json"

    if config_file.exists():
        print(f"Loading model from local cache: {local_path}")
        source = local_path
    else:
        print(f"Model not found locally. Downloading {model_name}...")
        source = model_name

    kwargs.setdefault("dtype", "auto")
    kwargs.setdefault("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)

    if source == model_name:
        print(f"Saving model to: {local_path}")
        local_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(local_path)

    return model


def load_tokenizer(model_name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    """
    Load tokenizer. Uses local cache if available,
    otherwise downloads from HuggingFace and saves locally.
    """
    if not model_dir.parent.exists():
        raise FileNotFoundError(
            f"Directory '{model_dir.parent}' does not exist.\n"
            f"You have to create it before continuing!"
        )

    local_path = model_dir / model_name
    tokenizer_file = local_path / "tokenizer_config.json"

    if tokenizer_file.exists():
        print(f"Loading tokenizer from local cache: {local_path}")
        source = local_path
    else:
        print(f"Tokenizer not found locally. Downloading {model_name}...")
        source = model_name

    tokenizer = AutoTokenizer.from_pretrained(source)

    if source == model_name:
        local_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_path)

    return tokenizer


def save_model(model, tokenizer, name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    """
    Save model and tokenizer to local directory.

    Args:
        model:      The model to save
        tokenizer:  The tokenizer to save
        name:       Name for the saved model (e.g. "Qwen/Qwen3-1.7B-Base-sft")
                    Saved to: /nobackup/<user>/models/Qwen/Qwen3-1.7B-Base-sft/
    """
    save_path = model_dir / name

    if save_path.exists():
        print(f"Warning: {save_path} already exists. Overwriting.")

    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)

    print(f"Saving tokenizer to: {save_path}")
    tokenizer.save_pretrained(save_path)

    print(f"Saved successfully: {save_path}")


def save_checkpoint(
    model, tokenizer, optimizer, name: str,
    global_step: int, examples_seen: int, best_accuracy: float,
    model_dir: Path = DEFAULT_MODEL_DIR,
):
    """Save model + training state for resuming."""
    save_model(model, tokenizer, name, model_dir)
    save_path = model_dir / name

    state = {
        "global_step": global_step,
        "examples_seen": examples_seen,
        "best_accuracy": best_accuracy,
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, save_path / "training_state.pt")
    print(f"  Training state saved to {save_path / 'training_state.pt'}")


def load_checkpoint(name: str, optimizer=None, model_dir: Path = DEFAULT_MODEL_DIR):
    """Load training state. Optionally restores optimizer."""
    state_path = model_dir / name / "training_state.pt"

    if not state_path.exists():
        return None

    state = torch.load(state_path)

    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
        print(f"  Optimizer state restored")
    else:
        print(f"  ⚠️ No optimizer state found, reinitializing optimizer")

    print(
        f"  Resumed: step={state['global_step']}, "
        f"examples={state['examples_seen']}, "
        f"best_acc={state['best_accuracy']:.1%}"
    )

    return state

def list_models(model_dir: Path = DEFAULT_MODEL_DIR) -> list[str]:
    """Return list of locally cached model names."""
    if not model_dir.exists():
        return []
    return sorted(
        str(cfg.parent.relative_to(model_dir))
        for cfg in model_dir.rglob("config.json")
    )


def download_model(model_name: str, model_dir: Path = DEFAULT_MODEL_DIR):
    """Download model + tokenizer to local cache, then free VRAM.
    All output is suppressed."""
    import contextlib
    import io
    import os
    import gc
    import logging

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    loggers = [logging.getLogger(n) for n in ("transformers", "huggingface_hub")]
    prev_levels = [(lgr, lgr.level) for lgr in loggers]
    for lgr in loggers:
        lgr.setLevel(logging.ERROR)

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model = load_model(model_name, model_dir)
            tokenizer = load_tokenizer(model_name, model_dir)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    finally:
        for lgr, lvl in prev_levels:
            lgr.setLevel(lvl)
        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)