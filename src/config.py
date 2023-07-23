from pathlib import Path

MAX_NUM_STORIES = 4  # TODO: change back to 100
MODEL = "gpt-3.5-turbo"  # "gpt-3.5-turbo" or "gpt-4" or "meta-llama/Llama-2-13b-chat-hf"
TEMPERATURE = 1
OUTPUT_DIR_PATH = Path("outputs")
STORIES_FILE_PATH = Path(f"{OUTPUT_DIR_PATH}/game_stories.json")
EVALUATION_FILE_PATH = Path(f"{OUTPUT_DIR_PATH}/evaluations.json")
SUMMARY_FILE_PATH = Path(f"{OUTPUT_DIR_PATH}/summary.json")