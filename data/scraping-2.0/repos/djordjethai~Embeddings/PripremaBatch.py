from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import os
from uuid import uuid4
import json
import datetime

# Import other necessary modules like UnstructuredFileLoader, RecursiveCharacterTextSplitter

def prepare_embeddings(chunk_size, chunk_overlap, folder_path, output):
    output_json_list = []

    # Iterate over all files in the specified folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            print(file_name)
            # Full path to the file
            file_path = os.path.join(folder_path, file_name)

            # Load the text file
            loader = UnstructuredFileLoader(file_path, encoding="utf-8")
            data = loader.load()

            # Split the document into smaller parts
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            texts = text_splitter.split_documents(data)

            # Process each chunk
            for i, document in enumerate(texts, 1):
                output_dict = {
                    "id": str(uuid4()),
                    "chunk": i,
                    "text": document.page_content,
                    "source": document.metadata.get("source", file_name),
                    "date": datetime.datetime.now().strftime("%d.%m.%Y")
                }

                output_json_list.append(output_dict)

    # Save all results in one JSON file
    output_file = os.path.join(folder_path, f'{output}.json')
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("[")  # Opening bracket with a newline
        for item in output_json_list[:-1]:
            # Format each dictionary as a string with a comma and newline
            item_string = json.dumps(item, ensure_ascii=False)
            file.write(item_string + ",\n")
        # Write the last item without a trailing comma
        file.write(json.dumps(output_json_list[-1], ensure_ascii=False))
        file.write("]")  # Closing bracket
    print(f"json file created {output_file}")
# Call the function with the desired parameters
prepare_embeddings(600, 0, "C:\\Users\\djordje\\Desktop\\Zapisnici\\Sredjen\\", "Kratki")
prepare_embeddings(2500, 0, "C:\\Users\\djordje\\Desktop\\Zapisnici\\Sredjen\\", "Dugacki")


# sta raditi 
# kreirati 2 json fajla: kratki i dugacki - ovde
# upsert kratki h ns1, dugacki h ns2, kratki s ns3, dugacki s ns4 - u app
# test