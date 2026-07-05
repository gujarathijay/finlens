"""
FinLens configuration.

Usage:
    from config.settings import settings
    print(settings.base_model_name)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model
    base_model_name: str = "google/gemma-2-9b-it"
    finetuned_model_path: str = str(MODELS_DIR / "finlens-lora")
    quantized_model_path: str = str(MODELS_DIR / "finlens-quantized")

    # Training
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 2048

    # Dataset
    dataset_path: str = str(DATA_DIR / "finlens_dataset.jsonl")
    val_split_ratio: float = 0.15

    # Inference
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    max_tokens: int = 1024
    temperature: float = 0.1

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    database_url: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'finlens.db'}"

    # Guardrails
    max_input_length: int = 8000
    confidence_threshold: float = 0.7

    # Monitoring
    otlp_endpoint: str = "http://localhost:4317"
    enable_tracing: bool = True

    # External
    hf_token: str = ""
    wandb_api_key: str = ""
    wandb_project: str = "finlens"


settings = Settings()