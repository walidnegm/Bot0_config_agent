SET THESE ENVIRONMENT VARIABLES
-------------------------------
export HF_TOKEN=your_token
export GITHUB_TOKEN=your_github_token
export OPENAI_API_KEY=your_openai_key

CREATE A VIRTUAL ENVIRONMNET & ACTIVATE
---------------------------------------
use requirements.txt

DOWNLOAD MOEDL TO USE LOCALLY
-------------------------------
huggingface-cli login
mkdir -p model/models--meta-llama--Meta-Llama-3-8B-Instruct
cd model/models--meta-llama--Meta-Llama-3-8B-Instruct

Run script to download the model - downloadllama.py

To Run the CLI
------------
cd ~/../Bot0_config_agent
python -m agent.cli --once "where are my model files" --openaicd ~/projects/Bot0_config_agent
