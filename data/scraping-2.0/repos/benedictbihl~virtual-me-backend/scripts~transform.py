import os
import json
import asyncio

from dotenv import load_dotenv

from langchain.document_transformers import DoctranQATransformer
from langchain.document_loaders import DirectoryLoader

load_dotenv()

# Define the folder where the wav files are located
root_folder = os.getenv("ROOT_DIR") or ""
print("Root folder: ", root_folder)

loader = DirectoryLoader(root_folder + "transcripts/", glob="**/*.txt")
documents = loader.load()
print("Loaded {} documents".format(len(documents)))


async def transform_documents():
    qa_transformer = DoctranQATransformer()
    print("Transforming documents...")
    transformed_documents = await qa_transformer.atransform_documents(documents)

    folder_path = os.path.join(root_folder, "transformed_documents")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for doc in transformed_documents:
        with open(
            os.path.join(
                folder_path,
                os.path.splitext(os.path.basename(doc.metadata["source"]))[0] + ".json",
            ),
            "w",
        ) as f:
            f.write(json.dumps(doc.metadata, indent=2))
    print("Done")


asyncio.run(transform_documents())
