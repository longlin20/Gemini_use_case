import re
import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts the text from a PDF file."""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text


def save_text_to_file(text: str, file_path: str) -> None:
    """Saves the text to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


def read_text_from_file(file_path: str) -> str:
    """Reads the text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def split_text_into_questions_and_solutions(text: str) -> tuple[str, str]:
    """Splits the text into questions and solutions."""
    pattern = re.compile(r'REFERENCE SOLUTIONS \(several solutions are possible\)', re.IGNORECASE)
    parts = pattern.split(text)
    questions = parts[0].strip() if len(parts) > 0 else ""
    solutions = parts[1].strip() if len(parts) > 1 else ""
    return questions, solutions


def remove_text_before_problem(input_file: str) -> None:
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if "Problem 1" in line:
            lines[i] = line[line.index("Problem 1"):]
            with open(input_file, 'w', encoding='utf-8') as file:
                file.writelines(lines[i:])
            break


def remove_university_header(file_path: str) -> None:
    """Removes the university header from a file."""
    pattern = re.compile(
        r'Universidad Carlos III de Madrid\s*Department\s*of Telematic Engineering\s*Systems Programming: English Group.*|\d+\s*$',
        re.MULTILINE)

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    modified_content = re.sub(pattern, '', content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(modified_content)

"""
Example usage:
year = '18-19'
all_items = os.listdir(f"{year}/exam")
files = [item for item in all_items if os.path.isfile(os.path.join(f"{year}/exam", item))]
for f in files:
    input_file_path = os.path.join(f"{year}/exam", f)
    text = extract_text_from_pdf(input_file_path)
    questions, solutions = split_text_into_questions_and_solutions(text)
    file_name_without_prefix = f.replace('.pdf', '')
    file_name_exam = f'parsed_exam/{file_name_without_prefix}_questions.txt'
    file_name_solutions = f'parsed_exam/{file_name_without_prefix}_solutions.txt'
    save_text_to_file(questions, file_name_exam)
    save_text_to_file(solutions, file_name_solutions)
    remove_text_before_problem(file_name_exam)
    remove_university_header(file_name_exam)
    remove_university_header(file_name_solutions)
"""