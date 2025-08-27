import os

from openai import OpenAI

openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_TOKEN"),
)
