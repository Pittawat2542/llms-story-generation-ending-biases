import json
import random
import re
import time

from transformers import Conversation
import openai

from src.obj_models import Story, Ending
from src.file_utils import save_story_obj_to_file, save_evaluation_to_file
from src.config import MODEL, TEMPERATURE


def get_chat_response(prompt: str, model=MODEL, converse_pipeline=None, temperature=TEMPERATURE) -> (str, str, float):
    if model != "gpt-3.5-turbo" and model != "gpt-4":
        print("Initiated chat with LLaMa-2.")
        conversation = Conversation(prompt)
        completion = converse_pipeline(conversation).generated_responses[-1]
        print("Completed chat with LLaMa-2.")
        return completion, model, -1
    else:
        print("Initiated chat with OpenAI API.")
        completion = openai.ChatCompletion.create(model=model,
                                                  temperature=temperature,
                                                  messages=[
                                                      {"role": "user", "content": prompt}])
        print("Completed chat with OpenAI API.")
        return completion.choices[0].message.content, model, temperature


def generate_game_story(converse_pipeline=None) -> Story:
    PROMPT = """Please write a brief 300-word game story synopsis with an ending. Please make sure to format your output as a code block using triple backticks (```json and ```).

Output format:
```json
{
"title": game title,
"story": game story synopsis until ending
}
```"""

    story_str, model, temp = get_chat_response(PROMPT, converse_pipeline=converse_pipeline)
    if "```json" in story_str:
        story_str = re.search(r"```json(.*)```", story_str, re.DOTALL).group(1).strip()
    if re.search(r"\{.*}", story_str, re.DOTALL) is None:
        story_str = f'{{"title": "N/A", "story": "{story_str}"}}'
    story_str = re.search(r"\{.*}", story_str, re.DOTALL).group(0).strip()
    stor_temp_obj = json.loads(story_str, strict=False)
    story_obj = Story(stor_temp_obj["title"], stor_temp_obj["story"], model, temp)

    save_story_obj_to_file(story_obj.to_json())

    return story_obj


def evaluate_game_story_ending(story: Story) -> Ending:
    PROMPT = f"""Please identify the type of ending in this story. Please make sure to format your output as a code block using triple backticks (```json and ```).

Title: {story.title}

Story:
{story.story}

Output format:
```json
{{ "ending": "positive", "negative", or "neutral" }}
```"""

    ending_str, model, temp = get_chat_response(PROMPT, temperature=0)
    if "```json" in ending_str:
        ending_str = re.search(r"```json(.*)```", ending_str, re.DOTALL).group(1).strip()
    if re.search(r"\{.*}", ending_str, re.DOTALL) is None:
        ending_str = f'{{"ending": "{ending_str}"}}'
    else:
        ending_str = re.search(r"\{.*}", ending_str, re.DOTALL).group(0).strip()
    ending_temp_obj = json.loads(ending_str, strict=False)
    ending_obj = Ending(ending_temp_obj["ending"], story.id, model, temp)

    save_evaluation_to_file(ending_obj.to_json())

    return ending_obj


def rate_limit_sleeper():
    if MODEL != 'gpt-3.5-turbo' and MODEL != 'gpt-4':
        return

    sleep_time = random.randint(3, 7)
    print(f"Sleeping for {sleep_time} seconds.")
    time.sleep(sleep_time)
