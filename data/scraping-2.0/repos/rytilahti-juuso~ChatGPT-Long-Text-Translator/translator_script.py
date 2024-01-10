# General imports
import os
import openai
import copy
# Similarity imports
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Set the environment variable
os.environ['API_KEY'] = 'PASTE_YOUR_API_KEY_HERE'

GPT_MODEL = "gpt-4"
"""
    If GPT model is 4, the similarity of successfully
    translated text chunk may be 1! While the GPT-3.5 was used
    this behaviour did not occur
"""

#Possible roles for individual messages:
#system
#user
#assistant



# return most recent chatGPT answer.
def get_translation_for_chunk(chunk,i, temperature=1, previous_messages=None):
    """
    returns the translated string
    """
    openai.api_key = os.getenv('API_KEY')
    
    all_messages = []
    # Add previous messages for chatGPT to use them as example.
    if(previous_messages):
        all_messages = all_messages + previous_messages
    
    all_messages.append({
                "role": "system", 
                "content": INITIAL_PROMPT
            })
    
    # Add new message to the end
    all_messages.append({
                "role": "user", 
                "content": chunk
            })
    
    print("API HAS BEEN CALLED!")
    # Call API
    chat_completion = openai.ChatCompletion.create(
        model=GPT_MODEL, 
        messages=all_messages,
        temperature= temperature,
        top_p=0.8
    )

    if 'choices' in chat_completion and chat_completion['choices']:
        #print(chat_completion)
        return chat_completion['choices'][0]['message']['content']
    else:
        return None

def create_messages(prompt, serverAnswer):
    return [
        {"role": "system", "content": INITIAL_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": serverAnswer}
    ]

def count_words(input_string):
    return len(input_string.split())

def split_into_chunks(input_string, chunk_size=240):
    """
    Args:
        input_string: Whole input string, should be in md-format
        chunk_size: Maximum size of a chunk in word count. 
                NOTE: If text is in middle of a code block when the chunk should be broken down, includes it
                    (so block's word count may exceed the set limit by a little bit)
    """
    # Store original text
    input_text = input_string

    # Split the input string by newline to get lines
    lines = input_string.split('\n')
    chunks = []
    current_chunk = []
    word_count = 0
    inside_code_block = False
    prev_line= ''
    for line in lines:
        line_word_count = count_words(line)
        # Check for code block start or end
        if line.startswith("```java"):
            inside_code_block = True
        elif "```" in prev_line and "```java" not in prev_line and inside_code_block:
            inside_code_block = False

        # If the addition of this line would exceed the chunk size, or
        # if it's a code block line, append the current chunk and start a new one
        #print(word_count+line_word_count)
        if word_count + line_word_count > chunk_size and not inside_code_block:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            word_count = line_word_count
        else:
            current_chunk.append(line)
            word_count += line_word_count
        prev_line = line

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def write_to_file(file_path, content):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print("Successfully wrote to the file: " + file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def read_from_file(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except IOError:
        print(f"Error reading the file: {file_path}")
        return None

def get_recursive_translation(chunk, i, previous_messages, attempt=1, max_attempts=8):
    """
    ChatGPT has tendency to (sometimes) repeat the previous translation again in the new message.
    If that happens, try to translate the chunk again.
    """
    trans = get_translation_for_chunk(chunk, i, previous_messages=copy.deepcopy(previous_messages))

    if not trans:
        return None
    
    write_to_file("./debug/chunk" + str(i) + "_original.md", chunk)
    write_to_file("./debug/chunk" + str(i) + "_translation.md", trans)
    
    if not previous_messages:
        return create_messages(chunk, trans)

    prev_translation = previous_messages[len(previous_messages) - 1]["content"]
    if are_texts_similar(prev_translation, trans) and attempt <= max_attempts:
        # If similar, recursively try again with varied output
        print("Trying a different translation...")
        return get_recursive_translation(chunk, i, previous_messages, attempt + 1)
    
    if attempt > max_attempts:
        # If still similar after max attempts, replace with the original
        print("The translation is still the same after multiple attempts, replacing the failed translation text with the text from original!")
        return create_messages(chunk, chunk)
    
    return create_messages(chunk, trans)


def get_messages(chunk, previous_messages, i):
    max_attempts =1  if GPT_MODEL == "gpt-4" else 8
    new_messages = get_recursive_translation(chunk, i, previous_messages, max_attempts=max_attempts)

    # Handle removing the oldest chat message if needed. Is trying to avoid overflow of context
    # one chunk is currently 3 messages (system, user, assistant) = in context are
    # currently 3 previous chunks
    if previous_messages and len(previous_messages) > 9:
        previous_messages = previous_messages[3:]

    if previous_messages:
        return previous_messages + new_messages
    else:
        return new_messages

def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the average of the last hidden state as the sentence representation
    return outputs.last_hidden_state.mean(dim=1).numpy()

def are_texts_similar(text1, text2, threshold=0.987):
    """
    ChatGPT 3.5 has tendency to repeat translation in previous 
    message instead of translating the new chunk.
    """
    vec1 = get_text_embedding(text1)
    vec2 = get_text_embedding(text2)
    
    similarity = cosine_similarity(vec1, vec2)[0][0]
    print("similarity is: " + similarity.astype(str))
    return similarity > threshold

LATEX_GENERAL_TRANSLATION_PROMPT = "You are a translator. Translate material in the latex file to English. Don't translate the comments. Do not alter the latex syntax, even if you deem it, for example, to miss some elements."
GENERAL_TRANSLATION_PROMPT_PLAIN_TEXT_AND_MD = "You are a translator. Translate the material to English." # Not thoroughly tested, but should work for basic usage.
TRANSLATE_AND_LOCALIZE_STUDY_MATERIAL_PROMPT_PLAIN_TEXT_OR_MD = "You are a translator. Localize and translate the study materials to English. Keep the meaning of the exercise in translation, but it does not need to be literal translation. If there are Finnish names change them to names used in England. Keep every actor the same."

# ------------ SET-UP ------------
# Set the initial prompt 
INITIAL_PROMPT = ""
# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#file_path = "./small_examples/small_example.md"
file_path = "input.md"
file_content = read_from_file(file_path)
# ---------------------------------

if not INITIAL_PROMPT:
    print("There seems to be some additional steps that you need to take.")
    print("1.) In code line 206, select one of the prompts as the initial prompt")
    print("2.) In the code line 228, set the chunk size to correct one.")
    print("3.) Run the program again.")
    print("Program terminating...")
    exit(1)

if file_content:
    USE_DEBUG_TEXT_IN_THE_OUTPUT = False
    CHUNK_SIZE_LATEX_GPT_4 = 240
    CHUNK_SIZE_PLAIN_TEXT_OR_MD_GPT_4 = 290
    chunks = split_into_chunks(file_content, chunk_size=CHUNK_SIZE_LATEX_GPT_4)
    final_text = ""
    previous_messages = None
    print("input.md has been broken down to "+str(len(chunks)) + " chunks.")
    for i, chunk in enumerate(chunks):
        print("    ")
        print("    ")
        print("Currently processing chunk " + str(i)+"/"+str(len(chunks)-1))
        messages = get_messages(chunk, previous_messages, i=i)
        
        #Include previous messages to the context
        if(previous_messages is None):
            previous_messages = messages
        else:
            #TODO take more messages to context if the word count is small enough
            # Currently takes only the latest message to account as context.
            previous_messages = messages
        # Latest element, value of content property
        trans = messages[len(messages)-1]["content"]
        
        #Divination between chuns to add readability (Normally woith GPT-3.5 if the translation fails, the translation of the whole chunk fails)
        chunk_divination = "\n\n---\n# Chunk "+ str(i)+"\n---\n\n" 
        if not USE_DEBUG_TEXT_IN_THE_OUTPUT:
            final_text = final_text + trans # exclude the debug text
        else:
            final_text =final_text + chunk_divination + trans
        # In case the translation fails to an error when only part of the translation is done
        # write the currently translated text also to the output.md
        write_to_file("output.md", final_text)    
    print("  ")
    print("  ")
    write_to_file("output.md", final_text)
