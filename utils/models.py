from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

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