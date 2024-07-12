import concurrent
import json
import os
import re
from typing import List

from langchain_google_vertexai import VertexAIEmbeddings
from pdf2image import convert_from_path
import base64
from io import BytesIO
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from tqdm import tqdm
import time
import pandas as pd

def convert_doc_to_images(path):
    images = convert_from_path(path)
    return images


def get_img_uri(img):
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image


def get_llm_config():
    system_prompt = '''
You will be provided with an image of a pdf page or a slide. Your goal is to extract the test-type exercises. Extract first the questions, then the possible answers, and finally the correct answer (write the full answer). The format should be as follows:

Question: [The question text]
Options: 
a. [Option A]
b. [Option B]
c. [Option C]
d. [Option D]
Answer: [The correct answer (write the full answer)]

Exclude elements that are not relevant to the content. Do not mention page numbers or the position of the elements on the image.
    '''
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
    }

    return system_prompt, generation_config


def analyze_image(data_uri, system_prompt, generation_config):
    vertexai.init(project="uc3m-it-gradient-h2olearn", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-pro-001",
    )

    image = Part.from_data(
        mime_type="image/jpeg",
        data=data_uri)

    responses = model.generate_content(
        [image, system_prompt],
        generation_config=generation_config
    )

    return responses.text


def analyze_doc_image(img, system_prompt, generation_config):
    img_uri = get_img_uri(img)
    data = analyze_image(img_uri, system_prompt, generation_config)
    return data


def process_docs(path, system_prompt, generation_config, json_path):
    imgs = convert_doc_to_images(path)
    pages_description = []

    doc = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:

        futures = []
        for img in imgs[1:]:
            futures.append(executor.submit(analyze_doc_image, img, system_prompt, generation_config))
            time.sleep(3)  # Adding a 3-second delay between requests

        with tqdm(total=len(imgs)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

        for f in futures:
            res = f.result()
            pages_description.append(res)

    doc['pages_description'] = pages_description

    with open(json_path, 'w') as f:
        json.dump(doc, f)


def process_questions_data(file):
    with open(file, 'r') as f:
        docs = json.load(f)

    questions_data = []
    for content in docs["pages_description"]:
        questions = re.findall(r"(?:Question:|Question\*\*:\*\*)\s*(.*?)\n(?:Options:|\*\*Options:\*\*)", content, re.DOTALL)
        options = re.findall(r"(?:Options:|\*\*Options:\*\*)\s*(.*?)\n(?:Answer:|\*\*Answer:\*\*)", content, re.DOTALL)
        answers = re.findall(r"(?:Answer:|\*\*Answer:\*\*)\s*(.*?)(?=\n|$)", content, re.DOTALL)

        for q, opt, ans in zip(questions, options, answers):
            opt_cleaned = re.sub(r'\n+', '\n', opt).strip()
            questions_data.append({
                "Question and Options": f"Question: {q.strip()}\nOptions:\n{opt_cleaned}",
                "Answer": ans.strip()
            })

    df = pd.DataFrame(questions_data)
    new_file = file.replace(".json", ".xlsx")
    df.to_excel(new_file, index=False)

"""
year = '18-19 (1)'
all_items = os.listdir(f"{year}/test")
files = [item for item in all_items if os.path.isfile(os.path.join(f"{year}/test", item))]
system_prompt, generation_config = get_llm_config()

for f in files:
    input_file_path = os.path.join(f"{year}/test", f)
    new_file = f.replace(".pdf", ".json")
    result_file_path = os.path.join("parsed_exam/test_for_context", f"{new_file}")
    process_docs(input_file_path, system_prompt, generation_config, result_file_path)
    process_questions_data(result_file_path)
"""