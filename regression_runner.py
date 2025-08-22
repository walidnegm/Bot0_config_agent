# regression_runner.py
import subprocess
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define regression scenarios (command split into tokens)
SCENARIOS = [
    # GPT
    ["python", "-m", "agent.cli", "--api-model", "gpt-4.1-mini", "--once",
     "Find all YAML files in the project and then create a summary of their contents."],
    ["python", "-m", "agent.cli", "--api-model", "gpt-4.1-mini", "--once",
     "List all files in the ./agent directory excluding __pycache__, .git, and venv."],
    ["python", "-m", "agent.cli", "--api-model", "gpt-4.1-mini", "--once",
     "is there a file called planner.py in the project"],
    ["python", "-m", "agent.cli", "--api-model", "gpt-4.1-mini", "--once",
     "count the files in the agent directory"],

    # Gemini
    ["python", "-m", "agent.cli", "--api-model", "gemini-1.5-flash-latest", "--once",
     "count the files in the agent directory"],
    ["python", "-m", "agent.cli", "--api-model", "gemini-1.5-flash-latest", "--once",
     "List all files in the ./agent directory excluding __pycache__, .git, and venv."],

    # LLaMA 2.7 GPTQ
    ["python", "-m", "agent.cli", "--local-model", "llama_2_7b_chat_gptq", "--once",
     "count the number of files in the tools folder"],

    # LLaMA 3.2 3B GPTQ
    ["python", "-m", "agent.cli", "--local-model", "llama_3_2_3b_gptq", "--once",
     "count the number of files in the agent folder"],
    ["python", "-m", "agent.cli", "--local-model", "llama_3_2_3b_gptq", "--once",
     "Find all YAML files in the project and then create a summary of their contents."],
]

def run_scenarios():
    if not SCENARIOS:
        logging.error("No scenarios defined.")
        sys.exit(1)

    for i, cmd in enumerate(SCENARIOS, start=1):
        logging.info("="*80)
        logging.info(f"Scenario {i}/{len(SCENARIOS)}")
        logging.info("Command: " + " ".join(cmd))

        # Extract the actual natural language instruction (after --once)
        try:
            instr_index = cmd.index("--once") + 1
            instruction = cmd[instr_index]
        except ValueError:
            instruction = "<no instruction found>"
        logging.info(f"📝 Instruction: {instruction}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            stdout_lines = result.stdout.strip().splitlines()
            stderr_lines = result.stderr.strip().splitlines()

            # Capture last several lines of stdout
            last_output = "\n".join(stdout_lines[-5:]) if stdout_lines else "<no stdout>"

            if result.returncode == 0:
                logging.info("✅ SUCCESS")
            else:
                logging.error(f"❌ FAILURE (exit code {result.returncode})")

            logging.info("📤 Last CLI output (tail):\n" + last_output)

            if stderr_lines:
                logging.warning("⚠️ stderr (tail):\n" + "\n".join(stderr_lines[-3:]))

        except Exception as e:
            logging.exception(f"Error running scenario {i}: {e}")

    logging.info("="*80)
    logging.info("📌 Reminder: Edit regression_runner.py to add/remove scenarios.")
    logging.info("Each entry in SCENARIOS is just a CLI command list.")

if __name__ == "__main__":
    run_scenarios()

