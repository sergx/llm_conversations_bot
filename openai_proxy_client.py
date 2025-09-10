import httpx
from openai import OpenAI
from config import SHARED_PROXY_URL, OPENAI_API_KEY

def openai_client():
    http_client = httpx.Client(
        proxy=SHARED_PROXY_URL,
        timeout=120)
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
    return client