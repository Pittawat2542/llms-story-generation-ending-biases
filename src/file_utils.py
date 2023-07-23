import json
import os

from src.config import OUTPUT_DIR_PATH, STORIES_FILE_PATH, EVALUATION_FILE_PATH, SUMMARY_FILE_PATH


def save_story_obj_to_file(story_json_obj: dict):
    if not os.path.isdir(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)
        print(f"Output directory {str(OUTPUT_DIR_PATH)} does not exist. Created new directory.")

    if not os.path.isfile(STORIES_FILE_PATH) or os.stat(STORIES_FILE_PATH).st_size == 0:
        with open(STORIES_FILE_PATH, "w") as f:
            file_obj = {"game_stories": [], "count": 0}
            json.dump(file_obj, f, indent=2)
            print(f"Output file {str(STORIES_FILE_PATH)} does not exist. Created new file.")

    with open(STORIES_FILE_PATH, "r+") as f:
        file_obj = json.load(f)
        file_obj["game_stories"].append(story_json_obj)
        file_obj["count"] += 1

        f.seek(0)
        json.dump(file_obj, f, indent=2)
        print(
            f"Saved story object with id {story_json_obj['id']} to {str(STORIES_FILE_PATH)}.")


def save_evaluation_to_file(ending_json_obj: dict):
    if not os.path.isdir(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)
        print(f"Output directory {str(OUTPUT_DIR_PATH)} does not exist. Created new directory.")

    if not os.path.isfile(EVALUATION_FILE_PATH) or os.stat(EVALUATION_FILE_PATH).st_size == 0:
        with open(EVALUATION_FILE_PATH, "w") as f:
            file_obj = {"evaluations": [], "count": 0}
            json.dump(file_obj, f, indent=2)
            print(f"Output file {str(EVALUATION_FILE_PATH)} does not exist. Created new file.")

    with open(EVALUATION_FILE_PATH, "r+") as f:
        file_obj = json.load(f)
        file_obj["evaluations"].append(ending_json_obj)
        file_obj["count"] += 1

        f.seek(0)
        json.dump(file_obj, f, indent=2)
        print(
            f"Saved evaluation object with id {ending_json_obj['id']} to {str(EVALUATION_FILE_PATH)}.")


def save_aggregated_results_to_file(ending_objs, story_objs):
    if not os.path.isdir(OUTPUT_DIR_PATH):
        os.mkdir(OUTPUT_DIR_PATH)
        print(f"Output directory {str(OUTPUT_DIR_PATH)} does not exist. Created new directory.")

    with open(SUMMARY_FILE_PATH, "w") as f:
        file_obj = {"summary": {
            "num_stories": len(story_objs),
            "num_endings": len(ending_objs),
            "num_positive_endings": len(
                [ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'positive']),
            "num_negative_endings": len(
                [ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'negative']),
            "num_neutral_endings": len(
                [ending_obj for ending_obj in ending_objs if ending_obj['ending'] == 'neutral']),
            "stories": [{
                "title": story_obj["title"],
                "story": story_obj["story"],
                "id": story_obj["id"],
                "model": story_obj["model"],
                "temperature": story_obj["temperature"],
                "ending": [ending_obj["ending"] for ending_obj in ending_objs if
                           ending_obj["story_id"] == story_obj["id"]][0]
            } for story_obj in story_objs],
        }}
        json.dump(file_obj, f, indent=2)
        print(f"Output file {str(SUMMARY_FILE_PATH)} does not exist. Created new file.")
