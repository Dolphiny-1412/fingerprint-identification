import os

# OpenAI API Configuration (load from environment; never commit secrets)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
