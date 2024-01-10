import os
import re
import PyPDF2
import openai
import dotenv
import os
import json

dotenv.load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


def get_exam_paths() -> list[str]:
    root_dir = "previous_exams"

    pdf_paths = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Check if the file is a PDF and follows the given naming patterns
            if file.endswith(".pdf") and ("_svar.pdf" not in file):
                pdf_paths.append(os.path.join(subdir, file))

    return pdf_paths


def extract_text_from_pdf(file_path: str) -> str:
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


def extract_questions_from_text(text: str) -> dict:
    # Regex pattern for "Opgave x" followed by question text
    pattern = r"\n([1-5]\.[\da-zA-Z])( \[\d{1,2}%\])?\s(.*?)(?=\n[1-5]\.[\da-zA-Z]|\d*Opga|2\s*$)"
    matches = re.findall(pattern, text, re.DOTALL)

    # Extract only the text part (exclude the x.y pattern) and remove leading/trailing spaces
    questions = {match[0]: match[2].strip() for match in matches}

    pattern_spg = r"(Spg\. [1-5][a-z])\s(.*?)(?=Spg\. [1-5][a-z]|\d*Opga|2\s*$)"
    matches_spg = re.findall(pattern_spg, text, re.DOTALL)

    for match in matches_spg:
        if match[0] not in questions:
            questions[match[0]] = match[1].strip()

    return questions


def uncorrupt_questions(questions: dict[str, str]) -> dict[str, str]:
    uncorrupt_questions = {}
    for key, text in questions.items():
        prompt = (
            f"Fix den følgende tekst på dansk. Kun skriv den fiksede tekst: \n{text}"
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )

        revised_text = completion.choices[0].message["content"]
        uncorrupt_questions[key] = revised_text

    return uncorrupt_questions


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path).split(".")[0]


# Write questions to file. Create new if file does not exist else append to existing file
def write_questions_to_file(
    key: str, questions: dict[str, str], file_path: str
) -> None:
    with open(file_path, "r+") as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            data = {}

        data[key] = questions

        file.seek(0)
        json.dump(data, file, indent=4)
        file.truncate()


# Check if key is in file
def key_in_file(key: str, file_path: str) -> bool:
    with open(file_path, "r") as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            data = {}

        return key in data


def main():
    pdf_paths = get_exam_paths()
    file_path = "questions.json"

    for pdf_path in pdf_paths:
        file_name = get_file_name(pdf_path)
        if key_in_file(file_name, file_path):
            continue
        text = extract_text_from_pdf(pdf_path)
        questions = extract_questions_from_text(text)
        questions = uncorrupt_questions(questions)

        write_questions_to_file(file_name, questions, f"questions.json")


if __name__ == "__main__":
    main()
