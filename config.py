
import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

COST_CACHE_FILE = "openai_cost_cache.json"
COST_CACHE_TTL = 24 * 3600  # 1 день


if os.getenv("DEVELOPER_CHAT_IDS"):
    DEVELOPER_CHAT_IDS = [int(_) for _ in os.getenv("DEVELOPER_CHAT_IDS").split(",")]
else:
    DEVELOPER_CHAT_IDS = []