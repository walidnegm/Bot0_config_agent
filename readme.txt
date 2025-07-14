export HF_TOKEN=your_token
export GITHUB_TOKEN=your_github_token
export OPENAI_API_KEY=your_openai_key
cd ~/../Bot0_config_agent
python -m agent.cli --once "where are my model files" --openaicd ~/projects/Bot0_config_agent
