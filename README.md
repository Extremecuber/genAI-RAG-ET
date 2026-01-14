Hello, This is my first ever GenAI project and its a basic RAG implementation to get started.

Follow these steps to run this in your own system.
## Setup

1. Install Python 3.10+
2. Create venv
3. pip install -r requirements.txt
4. Install Ollama
5. ollama pull llama3:8b
6. cp .env.example .env

## Run
python -m src.runner.app ingest
python -m src.runner.app query "Your question"

A lot more changes are coming as I am working on this project daily as of Jan second week. 
