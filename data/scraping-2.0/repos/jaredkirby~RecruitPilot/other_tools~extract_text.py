from langchain.document_loaders import PDFPlumberLoader
import os


def ingest_pdf(resume):
    print(f"Loading resume: {resume}...")
    loader = PDFPlumberLoader(resume)
    documents = loader.load()
    resume_text = "\n".join(
        [doc.page_content for doc in documents]
    )  # Concatenating the text content
    return resume_text


def main(resumes_directory, folder_path):
    for resume in folder_path:
        try:
            resume_path = os.path.join(resumes_directory, resume)
            print(f"Processing resume: {resume_path}")
            resume_text = ingest_pdf(resume_path)

            # Saving the extracted text to a file
            output_file_path = os.path.join(resumes_directory, f"{resume}_text.txt")
            with open(output_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(resume_text)

            print(f"Text content saved to {output_file_path}")
        except Exception as e:
            print(f"An error occurred while processing resume {resume}: {e}")
            continue


if __name__ == "__main__":
    resumes_directory = "good_res"
    folder_path = os.listdir(resumes_directory)
    main(resumes_directory, folder_path)
