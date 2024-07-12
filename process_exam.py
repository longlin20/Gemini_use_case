import os
import re
import PyPDF2


def extract_text_from_pdf(pdf_path):
    """Extrae el texto de un archivo PDF."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text_pypdf2 = ''
        for page in pdf_reader.pages:
            text_pypdf2 += page.extract_text()
    return text_pypdf2


def save_text_to_file(text, file_path):
    """Guarda el texto en un archivo de texto."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)


def read_text_from_file(file_path):
    """Lee el texto de un archivo de texto."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def split_text_into_questions_and_solutions(text):
    """Divide el texto en preguntas y soluciones usando una expresión regular."""
    patron_division = re.compile(r'REFERENCE SOLUTIONS \(several solutions are possible\)', re.IGNORECASE)
    partes = patron_division.split(text)
    preguntas = partes[0].strip() if len(partes) > 0 else ""
    soluciones_y_criterios = partes[1].strip() if len(partes) > 1 else ""
    return preguntas, soluciones_y_criterios


def remove_text_before_problem(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "Problem 1" in line:
            print("Problem 1 found at line", i)
            lines[i] = line[line.index("Problem 1"):]  # Keep only text starting from "Problem 1"
            with open(input_file, 'w', encoding='utf-8') as file:
                file.writelines(lines[i:])
            break


def remove_university_header(file_path):
    # Define a pattern to match the lines to be removed
    pattern = re.compile(
        r'Universidad Carlos III de Madrid\s*Department\s*of Telematic Engineering\s*Systems Programming: English Group.*|\d+\s*$',
        re.MULTILINE)

    # Read the original content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove the matching lines from the content
    modified_content = re.sub(pattern, '', content)

    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)


"""
year = '18-19'
all_items = os.listdir(f"{year}/exam")
files = [item for item in all_items if os.path.isfile(os.path.join(f"{year}/exam", item))]
for f in files:
    input_file_path = os.path.join(f"{year}/exam", f)
    text = extract_text_from_pdf(input_file_path)
    questions, answer_criteria = split_text_into_questions_and_solutions(text)
    file_name_without_prefix = f.replace('.pdf', '')
    file_name_exam = f'parsed_exam/{file_name_without_prefix}_questions.txt'
    file_name_answer = f'parsed_exam/{file_name_without_prefix}_answers_criterios.txt'
    save_text_to_file(questions, file_name_exam)
    save_text_to_file(answer_criteria, file_name_answer)
    remove_text_before_problem(file_name_exam)
    remove_university_header(file_name_exam)
    remove_university_header(file_name_answer)
"""