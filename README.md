# What Is Waiting for Us at the End? Inherent Biases of Game Story Endings in Large Language Models ([Paper](https://link.springer.com/chapter/10.1007/978-3-031-47658-7_26))

This repository contains the code for the paper "What Is Waiting for Us at the End? Inherent Biases of Game Story Endings in Large Language Models" accepted at [ICIDS 2023](http://icids2023.ardin.online).

## Authors
Pittawat Taveekitworachai, Febri Abdullah, Mustafa Can Gursesli, Mury F. Dewantoro, Siyuan Chen, Antonio Lanata, Andrea Guazzini, and Ruck Thawonmas

## Abstract

This study investigates biases present in large language models (LLMs) when utilized for narrative tasks, specifically in game story generation and story ending classification. Our experiment involves using popular LLMs, including GPT-3.5, GPT-4, and Llama 2, to generate game stories and classify their endings into three categories: positive, negative, and neutral. The results of our analysis reveal a notable bias towards positive-ending stories in the LLMs under examination. Moreover, we observe that GPT-4 and Llama 2 tend to classify stories into uninstructed categories, underscoring the critical importance of thoughtfully designing downstream systems that employ LLM-generated outputs. These findings provide a groundwork for the development of systems that incorporate LLMs in game story generation and classification. They also emphasize the necessity of being vigilant in addressing biases and improving system performance. By acknowledging and rectifying these biases, we can create more fair and accurate applications of LLMs in various narrative-based tasks.

## File structure
- `main.py`: The main script for story generation and evaluation using ChatGPT.
- `requirements.txt`: The requirements file for the project.
- `output/`: The directory containing the generated results (generated stories and evaluated endings) by ChatGPT.
- `src/`: The directory containing utility files for `main.py`

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n chatgpt-biases python=3.11
```
and activate it:
```bash
conda activate chatgpt-biases
```
1. Copy `.env.example` and rename it to `.env`. Follow instructions on [this page](https://platform.openai.com/docs/api-reference/authentication) to obtain your own OpenAI API key and on [this page](https://huggingface.co/docs/hub/security-tokens) for HuggingFace authentication token. You may also need to follow instructions on [this page](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to access Llama 2 models.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. Change configurations in `src/config.py` as needed.
4. Run the script for story generation and evaluation:
```bash
python main.py
```

## Citation
```bib
@InProceedings{10.1007/978-3-031-47658-7_26,
  author="Taveekitworachai, Pittawat and Abdullah, Febri and Gursesli, Mustafa Can and Dewantoro, Mury F. and Chen, Siyuan and Lanata, Antonio and Guazzini, Andrea and Thawonmas, Ruck",
  editor="Holloway-Attaway, Lissa and Murray, John T.",
  title={{"What Is Waiting for Us at the End? Inherent Biases of Game Story Endings in Large Language Models"}},
  booktitle="Interactive Storytelling",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="274--284",
  abstract="This study investigates biases present in large language models (LLMs) when utilized for narrative tasks, specifically in game story generation and story ending classification. Our experiment involves using popular LLMs, including GPT-3.5, GPT-4, and Llama 2, to generate game stories and classify their endings into three categories: positive, negative, and neutral. The results of our analysis reveal a notable bias towards positive-ending stories in the LLMs under examination. Moreover, we observe that GPT-4 and Llama 2 tend to classify stories into uninstructed categories, underscoring the critical importance of thoughtfully designing downstream systems that employ LLM-generated outputs. These findings provide a groundwork for the development of systems that incorporate LLMs in game story generation and classification. They also emphasize the necessity of being vigilant in addressing biases and improving system performance. By acknowledging and rectifying these biases, we can create more fair and accurate applications of LLMs in various narrative-based tasks.",
  isbn="978-3-031-47658-7"
}
```
