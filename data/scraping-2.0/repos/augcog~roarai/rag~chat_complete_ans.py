import os
import pickle
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = "empty"
openai.api_base = "http://localhost:8000/v1"

def wizard_coder(history: list[dict]):
    DEFAULT_SYSTEM_PROMPT = history[0]['content']+'\n\n'
    B_INST, E_INST = "### Instruction:\n", "\n\n### Response:\n"
    messages = history.copy()
    messages_list=[DEFAULT_SYSTEM_PROMPT]
    messages_list.extend([
        f"{B_INST}{(prompt['content']).strip()}{E_INST}{(answer['content']).strip()}\n\n"
        for prompt, answer in zip(messages[1::2], messages[2::2])
    ])
    messages_list.append(f"{B_INST}{(messages[-1]['content']).strip()}{E_INST}")
    return "".join(messages_list)
def chat_completion(system_message, human_message):
    history = [{"role": "system", "content": system_message}, {"role": "user", "content": human_message}]
    # if model=='local':
    #     prompt=wizard_coder(history)
    # elif model=='openai':
    #     prompt=gpt(history)
    # print(prompt)
    # completion = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo', messages=messages, temperature=0,
    # )
    prompt=wizard_coder(history)
    completion=openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=history, temperature=0, max_tokens=500)
    # completion = openai.Completion.create(model='gpt-3.5-turbo', prompt=prompt, temperature=0, max_tokens=500)
    # print(completion)
    # print(completion)

    # answer=completion['choices'][0]['message']["content"]
    answer=completion['choices'][0]['message']["content"]
    return answer


# Directory containing the pickle files
pickle_directory = 'question_set'

output_file_path = 'output_zephyr_short.txt'  # File where you want to save the output

# Open the output file
with open(output_file_path, 'w') as output_file:
    # Loop through all files in the directory
    for filename in os.listdir(pickle_directory):
        if filename.endswith('.pkl'):  # Check for .pkl extension
            # Construct the full path to the file
            file_path = os.path.join(pickle_directory, filename)

            # Open and load the pickle file contents
            with open(file_path, 'rb') as file:
                loaded_data = pickle.load(file)

                # Access the data from the loaded pickle file
                loaded_question = loaded_data.get('question')
                loaded_doc = loaded_data.get('doc')
                loaded_doc_id = loaded_data.get('doc_id')

                # Prepare the text to write to the output file
                file_output = f"File: {filename}\n"
                file_output += f"Question: {loaded_question}\n"
                # system_message = "Answer the question based on doc1, doc2 and doc3."
                system_message = "Answer the question based on doc1, doc2 and doc3. Make the answer short."
                doc = ""

                for i, docu in enumerate(loaded_doc):
                    doc += f"doc{i+1}: {docu}\n"
                file_output += f"Docs:\n {doc}\n"
                human_message = "question: " + loaded_question + "\n---\n" + doc

                answer = chat_completion(system_message, human_message)
                file_output += f"Answer: {answer}\n"
                answer = chat_completion("Answer the question. Mak the answer short", loaded_question)
                file_output += "=" * 40 + "\n"  # Separator for readability
                file_output += f"Answer without doc: {answer}\n"
                file_output += "-" * 40 + "\n"  # Separator for readability

                # Write to the output file
                output_file.write(file_output)
