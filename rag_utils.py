import json
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

CHROMA_PATH = Path("chroma")
EMBEDDING_CONFIG_PATH = CHROMA_PATH / "embedding_config.json"
HF_MODEL_NAME = "all-MiniLM-L6-v2"

def load_environment() -> None:
    load_dotenv()

def _build_hf_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

def get_embeddings():
    load_environment()
    return _build_hf_embeddings(), "huggingface"

def get_chat_model():
    load_environment()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

def save_embedding_config(provider: str) -> None:
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    EMBEDDING_CONFIG_PATH.write_text(
        json.dumps({"provider": provider}, indent=2),
        encoding="utf-8",
    )
