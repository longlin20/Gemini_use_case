import concurrent.futures
import json
import re
from io import BytesIO

import pandas as pd
from pdf2image import convert_from_path
import base64

from tqdm import tqdm
import time

from vertexai.generative_models import GenerativeModel, Part
from vertexai.init import init


def convert_document_to_images(file_path):
    return convert_from_path(file_path)


def get_image_data_uri(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image


def get_llm_config():
    system_prompt = (
        "You will be provided with an image of a pdf page or a slide. "
        "Your goal is to extract the test-type exercises. "
        "Extract first the questions, then the possible answers, "
        "and finally the correct answer (write the full answer). "
        "The format should be as follows:\n\n"
        "Question: [The question text]\n"
        "Options:\n"
        "a. [Option A]\n"
        "b. [Option B]\n"
        "c. [Option C]\n"
        "d. [Option D]\n"
        "Answer: [The correct answer (write the full answer)]\n\n"
        "Exclude elements that are not relevant to the content. "
        "Do not mention page numbers or the position of the elements on the image."
    )
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
    }
    return system_prompt, generation_config


def analyze_image(data_uri, system_prompt, generation_config):
    init(project="uc3m-it-gradient-h2olearn", location="us-central1")
    model = GenerativeModel("gemini-1.5-pro-001")
    image = Part.from_data(mime_type="image/jpeg", data=data_uri)
    responses = model.generate_content(
        [image, system_prompt], generation_config=generation_config
    )
    return responses.text


def analyze_document_image(image, system_prompt, generation_config):
    image_data_uri = get_image_data_uri(image)
    data = analyze_image(image_data_uri, system_prompt, generation_config)
    return data


def process_document(file_path, system_prompt, generation_config, json_file_path):
    images = convert_document_to_images(file_path)
    pages_description = []
    doc = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for img in images[1:]:
            futures.append(
                executor.submit(
                    analyze_document_image, img, system_prompt, generation_config
                )
            )
            time.sleep(3)
        with tqdm(total=len(images)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
        for f in futures:
            res = f.result()
            pages_description.append(res)
    doc["pages_description"] = pages_description
    with open(json_file_path, "w") as f:
        json.dump(doc, f)


def process_question_data(file_path):
    with open(file_path, "r") as f:
        docs = json.load(f)
    questions_data = []
    for content in docs["pages_description"]:
        questions = re.findall(
            r"(?:Question:|Question\*\*:\*\*)\s*(.*?)\n(?:Options:|\*\*Options:\*\*)",
            content, re.DOTALL
        )
        options = re.findall(
            r"(?:Options:|\*\*Options:\*\*)\s*(.*?)\n(?:Answer:|\*\*Answer:\*\*)",
            content,
            re.DOTALL,
        )
        answers = re.findall(
            r"(?:Answer:|\*\*Answer:\*\*)\s*(.*?)(?=\n|$)",
            content,
            re.DOTALL,
        )

        for question, option, answer in zip(questions, options, answers):
            cleaned_options = re.sub(r"\n+", "\n", option).strip()
            questions_data.append({
                "Question and Options": f"Question: {question.strip()}\nOptions:\n{cleaned_options}",
                "Answer": answer.strip(),
            })

    df = pd.DataFrame(questions_data)
    output_file_path = file_path.replace(".json", ".xlsx")
    df.to_excel(output_file_path, index=False)




# Example usage:
"""
year = '18-19 (1)'
all_items = os.listdir(f"{year}/test")
files = [item for item in all_items if os.path.isfile(os.path.join(f"{year}/test", item))]
system_prompt, generation_config = get_llm_config()

for file in files:
    input_file_path = os.path.join(f"{year}/test", file)
    new_file = file.replace(".pdf", ".json")
    result_file_path = os.path.join("parsed_exam/test_for_context", new_file)
    process_document(input_file_path, system_prompt, generation_config, result_file_path)
    process_question_data(result_file_path)
"""
