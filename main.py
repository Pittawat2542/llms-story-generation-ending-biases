import os
import json

from dotenv import load_dotenv
import openai

from transformers import pipeline
from src.config import MODEL, TEMPERATURE, MAX_NUM_STORIES, STORIES_FILE_PATH, EVALUATION_FILE_PATH
from src.obj_models import Story
from src.file_utils import save_aggregated_results_to_file
from src.models import generate_game_story, evaluate_game_story_ending, rate_limit_sleeper


def main():
    converse = None
    if MODEL != "gpt-3.5-turbo" and MODEL != "gpt-4":
        hf_auth_token = os.getenv("HF_AUTH_TOKEN")

        if hf_auth_token is None:
            raise ValueError("HF_AUTH_TOKEN is not set.")

        converse = pipeline("conversational", model=MODEL, use_auth_token=hf_auth_token, temperature=TEMPERATURE,
                            max_length=4096)

    current_count = 0
    if os.path.isfile(STORIES_FILE_PATH):
        with open(STORIES_FILE_PATH, "r") as f:
            current_count = json.load(f)["count"]

    print("=== Generating game stories ===")
    print(f"Model: {MODEL}, Temperature: {TEMPERATURE}")
    while current_count < MAX_NUM_STORIES:
        print(f"--- Progress: {current_count}/{MAX_NUM_STORIES} ---")
        generate_game_story(converse_pipeline=converse)
        current_count += 1
        rate_limit_sleeper()

    print("Finished generating game stories.")

    print("=== Evaluating game story endings ===")
    print(f"Model: {MODEL}, Temperature: 0")

    with open(STORIES_FILE_PATH, "r") as f:
        story_objs = json.load(f)["game_stories"]
        if os.path.isfile(EVALUATION_FILE_PATH):
            story_objs = [story_obj for story_obj in story_objs if
                          story_obj["id"] not in [ending_obj["story_id"] for ending_obj in
                                                  json.load(open(EVALUATION_FILE_PATH, "r"))["evaluations"]]]

        print(f"Number of stories to evaluate: {len(story_objs)}")
        for story_obj in story_objs:
            print(f"--- Progress: {story_objs.index(story_obj)}/{len(story_objs)} ---")
            story = Story('', '')
            story.from_json(story_obj)
            evaluate_game_story_ending(story)
            rate_limit_sleeper()

    print("Finished evaluating game story endings.")

    print("=== Result Summarization ===")
    with open(EVALUATION_FILE_PATH, "r") as f:
        with open(STORIES_FILE_PATH, "r") as f2:
            story_objs = json.load(f2)["game_stories"]
            ending_objs = json.load(f)["evaluations"]
            print(f"Number of stories evaluated: {len(story_objs)}")
            print(f"Number of endings evaluated: {len(ending_objs)}")
            print(
                f"Number of stories with positive endings: {len([ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'positive'])}")
            print(
                f"Number of stories with negative endings: {len([ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'negative'])}")
            print(
                f"Number of stories with neutral endings: {len([ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'neutral'])}")

            save_aggregated_results_to_file(ending_objs, story_objs)


if __name__ == '__main__':
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    main()
