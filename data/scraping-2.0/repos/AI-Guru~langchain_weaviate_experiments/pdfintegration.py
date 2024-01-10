import dotenv
import os
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
import weaviate


# Load and validate the config.
config = dotenv.dotenv_values()
required_keys = [
    "OPENAI_API_KEY",
    "PDF_DATABASE_CLASS",
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
]
for key in required_keys:
    assert key in config, f"Key {key} not found in .env file"


def main():

    # Ask the user what to do.
    print("What do you want to do?")
    print("1. Create a database")
    print("2. Import a PDF")
    print("3. Ask a question")
    choice = input("Enter your choice: ")

    if choice == "1":
        create_database()

    if choice == "2":
        #pdf_path = input("Enter the path to the PDF: ")
        pdf_path = "/Users/tristanbehrens/CalibreCollection/John Romero/Doom Guy_ Life in First Person (322)/Doom Guy_ Life in First Person - John Romero.pdf"
        import_pdf(pdf_path)

    if choice == "3":
        question = input("Enter your question: ")
        #question = "What is John Romero's opinion about John Carmack?"
        ask_question(question)


def create_database():
    client = get_client()

    database_class = config["PDF_DATABASE_CLASS"]

    class_object = {
        "class": database_class,
        "description": "Pages from books",
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "model": "ada",
                "modelVersion": "002",
                "type": "text"
            }
        },
    }

    try:
        client.schema.create_class(class_object)
        print(f"Created class {class_object['class']}")
    except Exception as e:
        print(e)


def import_pdf(pdf_path):

    assert os.path.exists(pdf_path), f"PDF path does not exist: {pdf_path}"

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    for page in pages:
        #print(page.metadata)
        print(page.metadata.keys())
        #print(page.page_content)
        break

    client = get_client()
    database_class = config["PDF_DATABASE_CLASS"]

    with client.batch(
            batch_size=100
        ) as batch:
            
            # Batch import the papers.
            for i, page in enumerate(pages):
                print(f"Importing page: {i+1}/{len(pages)}")

                properties = {
                    "source": page.metadata["source"],
                    "page": page.metadata["page"] + 1,
                    "page_content": page.page_content,
                }

                batch.add_data_object(
                    properties,
                    database_class,
                )
                

# Ask a question.
def ask_question(question, limit=5):
    client = get_client()
    database_class = config["PDF_DATABASE_CLASS"]

    nearText = {
        "concepts": [question]
    }
    print(f"Querying concept: {question}")

    response = (
        client.query
        .get(database_class, ["source", "page", "page_content"])
        .with_near_text(nearText)
        .with_limit(limit)
        .do()
    )
    if "errors" in response:
        print(response["errors"])
        return

    results = response["data"]["Get"][database_class]
    print(f"Number of results: {len(results)}")

    # Check if everything is okay.
    for result in results:
        assert "page" in result, f"Title not found in result. Got keys: {result.keys()}"
        assert "page_content" in result, f"Authors not found in result. Got keys: {result.keys()}"
        assert "source" in result, f"Authors not found in result. Got keys: {result.keys()}"

    # Create the prompt.
    prompt = ""
    for result in results:
        page = result["page"]
        source = result["source"]
        page_content = result["page_content"]
        page_prompt = PromptTemplate.from_template("This is page {page} from {source}:\n\n{page_content}\n\n")
        page_prompt = page_prompt.format(page=page, source=source, page_content=page_content)
        prompt += page_prompt
    prompt += f"Please answer this question and tell me on which page you got the answer: {question}"
    prompt = PromptTemplate.from_template(prompt)

    # Run the model.
    model = ChatOpenAI(
        openai_api_key=config["OPENAI_API_KEY"],
    )
    assert model is not None
    chain = LLMChain(llm=model, prompt=prompt, verbose=False)
    result = chain.run(text="")
    assert isinstance(result, str), f"result is not a string: {type(result)}"

    print(result)


def get_client():
    client = weaviate.Client(
        url = config["WEAVIATE_URL"],  
        auth_client_secret=weaviate.AuthApiKey(api_key=config["WEAVIATE_API_KEY"]), 
        additional_headers = {
            "X-Openai-Api-Key": config["OPENAI_API_KEY"],
        })
    return client


if __name__ == "__main__":
    main()