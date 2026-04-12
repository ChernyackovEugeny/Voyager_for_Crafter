import os
from dotenv import load_dotenv

load_dotenv()

# --- DeepSeek API ---
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
DEEPSEEK_CODEGEN_MODEL: str = "deepseek-chat"  # DeepSeek V3

# --- CodeGen hyperparameters ---
CODEGEN_TEMPERATURE: float = 0.0
CODEGEN_MAX_FIX_ATTEMPTS: int = 3

if not DEEPSEEK_API_KEY:
    raise EnvironmentError(
        "DEEPSEEK_API_KEY is not set. "
        "Add it to .env at the project root."
    )
