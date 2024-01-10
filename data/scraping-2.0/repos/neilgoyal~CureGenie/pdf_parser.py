from langchain.document_loaders import PyMuPDFLoader
import ray
import openai

ray.init(runtime_env={"pip":["langchain","pymupdf","openai","transformers"]})

def parse_pdf(path_to_pdf: str) -> str:
    """Parse a PDF file and return the summarized text."""

    loader = PyMuPDFLoader(path_to_pdf)
    data = loader.load()
    blood_work = [data[i].page_content for i in range(len(data))]
    # concat all pages into one string
    blood_work = " ".join(blood_work)
    messages_to_api = [
        {"role": "system", "content": "You are a highly intelligent assistant and health care professional."},
        {"role": "user", "content": "This is a patient's blood test result. Summarize with key stats and highlight any anomaly. Be concise, and ignore data that is within the normal range. " + blood_work}
    ]

    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=messages_to_api,
    temperature=0,
    )
    summary = response.choices[0]["message"]["content"]
    ray.shutdown()
    return summary

if __name__ == "__main__":
    path_to_pdf = "test_data_cp.pdf"
    summary = parse_pdf(path_to_pdf)
    print(summary)
    
