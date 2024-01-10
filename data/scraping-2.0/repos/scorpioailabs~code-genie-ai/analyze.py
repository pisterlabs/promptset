import os
import openai
import re
from loguru import logger
from pyAn.analyzer import CallGraphVisitor
from dotenv import load_dotenv
from supabase import create_client
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import ast 
from datetime import datetime, timedelta
import time
import difflib
from suggestion import Suggestion
# from langchain.llms import OpenAI
# from langchain import PromptTemplate

# Load the environment variables
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")
supabase = create_client(supabase_url, supabase_key)

model = os.environ.get("MODEL")

def generate_dependency_graph(file_list):
    dependencies = {}
    for file_str in file_list:
        file_obj = json.loads(file_str)
        file_path = os.path.normpath(file_obj['source'])
        logger.info(f"Processing file: {file_path}")
        if file_path.endswith(".py"):
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependencies.setdefault(file_path, set()).add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            dependencies.setdefault(file_path, set()).add(f"{module}.{alias.name}")
        else:
            print(f"Skipping non-Python file: {file_path}")
    return dependencies

def get_window_size(model):
    if model == 'gpt3':
        return 3000
    elif model == 'gpt4':
        return 7000
    else:
        raise ValueError("Invalid model specified.")
    
def check_elapsed_time(remaining_time, suggestions):
    return remaining_time <= 0 and len(suggestions) > 0

def check_rate_limit(user_request_timestamps, user_requests_hourly, start_time):
    current_time = datetime.now() 
    user_request_timestamps = [t for t in user_request_timestamps if current_time - t < timedelta(minutes=1)]
    user_requests_per_minute = len(user_request_timestamps)
    if user_requests_per_minute >= 20:
        wait_time = 60 - (current_time - user_request_timestamps[0]).total_seconds()
        logger.info(f"Reached maximum requests per minute. Waiting for {wait_time:.2f} seconds...")
        time.sleep(wait_time)

    user_request_timestamps.append(current_time)
    user_requests_hourly += 1
    if user_requests_hourly > 500:
        logger.warning("Reached maximum requests per hour. Aborting...")
        return False

    return True

def process_code(embedding, window_size, start_time, suggestions, dependencies_dict, embeddings_dict):
    code_content = embedding['content'].strip().lstrip('\ufeff')
    success = False
    file_path = os.path.normpath(embedding['metadata']['source'])

    # Log the number of requests being made
    requests_made = 0

    # Find dependencies based on the dependency graph
    dependencies = dependencies_dict.get(file_path, [])

    # Combine the code content with its dependencies
    combined_code = code_content
    for dependency in dependencies:
        dep_embedding = embeddings_dict.get(json.dumps(dependency))
        if dep_embedding:
            combined_code += f"\n\n### {dep_embedding['metadata']} ###\n{dep_embedding['content']}"

    # Process the combined_code with windowing
    for idx in range(0, len(combined_code), window_size):
        windowed_code = combined_code[idx:idx+window_size]

        # Add the windowed_code to the prompt
        prompt = """
            You are CodeGenie AI, a superintelligent AI that analyzes codebases and provides suggestions to improve or refactor code based on its underlying functionality. You are helpful, friendly, incredibly intelligent, and take pride in reviewing code and ensuring code readability.

            Analyze the code file and its dependencies, and suggest improvements based on code optimization, best practices, and opportunities for refactoring:

            {windowed_code}

            [END OF CODE FILE(S)]

            When providing suggestions, consider the following conditions and respond with 'No improvements to be made.' (AND NOTHING ELSE) if the code matches any of them:
            1. Configuration files
            2. Gitignore files
            3. Language-specific configuration files
            4. Obvious program-dependent config files
            5. Files with no obvious improvements

            However, if the code does not meet any of these conditions, provide helpful and friendly suggestions IN natural language to improve the code based on its underlying functionality. 
            It's really important to only enumerate suggestions at this point rather than code. Your role is to be helpful and friendly, so make sure your suggestions align with the given conditions and are genuinely useful when applicable.
        """

        chat = ChatOpenAI(
            streaming=False,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            temperature=0.5,
            request_timeout=60)
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

        # TODO - Add in support for api completion when it is released
        # llm = OpenAI(model_name=model, temperature=0.5, max_tokens=GetMaxTokensForModel(model))
        # prompt_template = PromptTemplate(input_variables=["windowed_code"], template=prompt)
        # prompt_template.format(windowed_code=windowed_code)

        chain = LLMChain(llm=chat, prompt=chat_prompt)
        logger.info(f"Running code analysis on file {embedding['metadata']}...")
        
        try:
            # Check elapsed time again and break the loop if it exceeds 60 seconds and there are enough suggestions
            # Calculate the remaining time
            remaining_time = 60 - (time.time() - start_time)
            if check_elapsed_time(remaining_time, suggestions):
                logger.warning("Rate limit error: 60 second limit reached. Terminating code analysis.")
                break
            
            response = chain.run(windowed_code=windowed_code)
            response_text = response.strip()
            # Increment the number of requests made
            requests_made += 1

            # Check if the response does not contain the specific phrase "No improvements to be made" as a substring
            if "No improvements to be made" not in response_text:
                similarity_threshold = 0.7
                
                # Check if there are any suggestions and calculate the Jaccard similarity
                if suggestions:
                    similarities = [jaccard_similarity(suggestion.suggestion, response_text) for suggestion in suggestions]
                    # log
                    logger.info(f"Similarities: {similarities}")
                    is_duplicate = any(similarity >= similarity_threshold for similarity in similarities)

                    if is_duplicate:
                        logger.info(f"Duplicate suggestion for file {embedding['metadata']}: {response_text}")
                        success = True
                else:
                    is_duplicate = False
                    
                # Add a new suggestion if there are no duplicates
                if not is_duplicate:
                    s = Suggestion(file=str(embedding['metadata']), suggestion=response_text)
                    logger.info(f"Adding suggestion for file {embedding['metadata']}: {response_text}")
                    suggestions.add(s)
                    logger.info(f"Suggested improvement for file {embedding['metadata']}: {response_text}")
                
                success = True  # Set success to True if the response is processed without errors

            else:
                logger.info(f"No improvements to be made for file {embedding['metadata']}")
                success = True  # Set success to True if the response is processed without errors
                continue
        
        except Exception as e:
            # Handle error from langchain
            logger.error(f"Error: {e}")
            # If the error is a rate limit error, retry the request
            if "RateLimitError" in str(e):
                wait_time = 20 * (2 ** retries)
                # Make sure we haven't reached the 60 second limit
                remaining_time = 60 - (time.time() - start_time)
                if check_elapsed_time(remaining_time, suggestions):
                    logger.warning("Rate limit error: 60 second limit reached. Terminating code analysis.")
                    break
                else:
                    logger.warning(f"Rate limit error: {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                    if retries >= 5:  # Break the loop if too many retries
                        logger.warning("Too many retries. Skipping the current file.")
                        success = True  # Set success to True to move to the next embedding

    # return success and suggestions
    return success, suggestions, requests_made

def analyze_and_suggest_improvements(batch_size=100, model='gpt4'):
    suggestions = set() # Initialize an empty set to store unique suggestions
    window_size = get_window_size(model)

    # Initialize counters for rate restrictions
    user_requests_hourly = 0
    user_request_timestamps = []

    start_time = time.time()

    logger.info("Started analyzing codebase.")

    while True:
        # Calculate the remaining time
        remaining_time = 60 - (time.time() - start_time)

        if check_elapsed_time(remaining_time, suggestions):
            logger.warning("Rate limit error: 60 second limit reached. Terminating code analysis.")
            break

        embeddings_query = (
            supabase
            .from_(os.environ.get("TABLE_NAME"))
            .select("metadata,content")
            .limit(batch_size)
        )

        result = embeddings_query.execute()

        if len(result.data) == 0:
            logger.info("No data to fetch.")
            break

        embeddings_list = result.data
        embeddings_dict = {json.dumps(embedding['metadata']): embedding for embedding in embeddings_list}
        file_list = [json.dumps(embedding['metadata']) for embedding in embeddings_list]

        dependencies_dict = generate_dependency_graph(file_list)

        for embedding in embeddings_list:
            success = False  # Add a flag to check if the request was successful
            while not success:  # Continue processing the current embedding until success or too many retries or rate limit reached or 1 minute elapsed
                # Check elapsed time again and break the loop if it exceeds 1 minute and there are enough suggestions
                remaining_time = 60 - (time.time() - start_time)
                if check_elapsed_time(remaining_time, suggestions):
                    logger.warning("Rate limit error: 60 second limit reached. Terminating code analysis.")
                    break

                # Check and enforce user rate restrictions
                if not check_rate_limit(user_request_timestamps, user_requests_hourly, start_time):
                    return

                success, new_suggestions, user_requests = process_code(embedding, window_size, start_time, suggestions, dependencies_dict, embeddings_dict)
                user_requests_hourly += user_requests
                user_request_timestamps.append(datetime.now()) 
                if new_suggestions is not None:
                    for suggestion in new_suggestions:
                        # Use the custom Suggestion class to create a hashable object
                        s = Suggestion(file=suggestion.file, suggestion=suggestion.suggestion)
                        # Add the suggestion to the set (only if it's unique)
                        suggestions.add(s)

        # Log the number of suggestions generated

        suggestions_list = [{'file': s.file, 'suggestion': s.suggestion} for s in suggestions]
        print(f"Generated {len(suggestions_list)} suggestions in total.")

        # Log the number of requests made
        print(f"Made {user_requests_hourly} requests in total.")

        implement_suggestions(suggestions_list)

def implement_suggestions(suggestions):
    write_suggestions_to_file(suggestions)
    
    for suggestion in suggestions:
        try:
            file_path, dictionary = get_file_path_and_dictionary(suggestion)
        except ValueError as e:
            print(e)
            continue

        try:
            original_code = read_original_code(file_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        updated_code = original_code
        apply_suggestion(updated_code, suggestion, file_path)

def get_file_path_and_dictionary(suggestion):
    s = suggestion['file']
    try:
        dictionary = ast.literal_eval(s)
    except ValueError as e:
        raise ValueError(f"Error parsing dictionary from suggestion: {e}")

    file_path = dictionary['source']
    return file_path, dictionary

def read_original_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        original_code = f.read()
    return original_code

def apply_suggestion(original_code, suggestion, file_path):
    # Process the original_code with windowing
    window_size = get_window_size(model)
    anchor_map = {}
    all_refactored_lines = []

    for idx in range(0, len(original_code), window_size):
        windowed_code = original_code[idx: idx + window_size]

        # Add the windowed_code to the prompt
        prompt= """
            You are CodeGenie AI. You are a superintelligent AI that refactors codebases based on suggestions provided by your counterpart AI to improve underlying functionality.
            You are:
            - helpful & friendly
            - incredibly intelligent
            - an uber developer who takes pride in writing code
            Utilize the following suggestion that your counterpart provided and implement them in the snippet we provide you.
            First you get the suggestion, then you get the current code window.
            Write code and nothing else, if you are writing anything in natural language then it will only be comments denoted by the appropriate syntax for the language you are writing in.
            Ultimately your goal is to implement the suggestions provided directly as code, be careful not to delete any code that is not part of the suggestions.
            Suggestions (in natural language):
            {suggestion_text}
            Code:
            {windowed_code}
            [END OF CODE FILE(S)]
        """
                     
        chat = ChatOpenAI(
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=True,
            temperature=0.5,
            request_timeout=60)
        system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
        chain = LLMChain(llm=chat, prompt=chat_prompt)
        logger.info(f"Implementing suggested improvements to code in {file_path}...")
        try:
            response = chain.run(suggestion_text=suggestion, windowed_code=windowed_code)

            # Get the refactored lines from the response
            original_lines = windowed_code.splitlines()
            refactored_lines = response.splitlines()
            all_refactored_lines.extend(refactored_lines)

            # Update the anchor_map with the untouched lines
            untouched_lines = find_untouched_lines(original_lines, refactored_lines)

            if not untouched_lines:
                print("No untouched lines found")

            for original_line_number, refactored_line_number in untouched_lines.items():
                if original_line_number < 0 or refactored_line_number < 0:
                    print(f"Skipping negative line numbers: original={original_line_number}, refactored={refactored_line_number}")
                    continue

                max_line_number = max(len(original_lines), len(refactored_lines))
                if original_line_number >= max_line_number or refactored_line_number >= max_line_number:
                    print(f"Skipping out-of-range line numbers: original={original_line_number}, refactored={refactored_line_number}")
                    continue

                anchor_map[idx + original_line_number] = idx + refactored_line_number
            
            # Merge the refactored code with the original code using the anchor_map
            merged_code = merge_code(original_code.splitlines(), all_refactored_lines, anchor_map)
            
            # Save the new file content to the file
            with open(file_path, 'w') as f:
                f.write(merged_code)
        except Exception as e:
            logger.error(f"Error: {e}")
            continue

def preprocess_lines(lines):
    pattern = re.compile(r"^\s*(#.*|\s*)$")
    preprocessed_lines = [line for line in lines.splitlines() if not pattern.match(line)]
    return preprocessed_lines

def find_untouched_lines(original_lines, refactored_lines):
    if not original_lines or not isinstance(original_lines, str):
        print("Invalid input for original_lines")
        return {}
    
    if not refactored_lines or not isinstance(refactored_lines, str):
        print("Invalid input for refactored_lines")
        return {}

    original_lines_list = preprocess_lines(original_lines)
    refactored_lines_list = preprocess_lines(refactored_lines)

    if not original_lines_list or not refactored_lines_list:
        print("No lines found in input")
        return {}

    matcher = difflib.SequenceMatcher(None, original_lines_list, refactored_lines_list)

    untouched_lines = {}
    for original_index, refactored_index, length in matcher.get_matching_blocks():
        if length > 0:
            for i in range(length):
                untouched_lines[original_index + i] = refactored_index + i

    return untouched_lines

def merge_code(original_lines, refactored_lines, anchor_map):
    merged_lines = []
    refactored_line_idx = 0

    for original_line_idx, original_line in enumerate(original_lines):
        if original_line_idx in anchor_map:
            # If the original line should remain unchanged, add it to the merged_lines
            merged_lines.append(original_line)
        else:
            # Check if refactored_line_idx is within the valid range
            if refactored_line_idx < len(refactored_lines):
                # Otherwise, add the corresponding refactored line to the merged_lines
                merged_lines.append(refactored_lines[refactored_line_idx])
                refactored_line_idx += 1
            else:
                print(f"Warning: refactored_line_idx out of range: {refactored_line_idx}")


    # Add any remaining refactored lines to the merged_lines that were not added during the loop
    while refactored_line_idx < len(refactored_lines):
        merged_lines.append(refactored_lines[refactored_line_idx])
        refactored_line_idx += 1

    return "\n".join(merged_lines)

def write_suggestions_to_file(suggestions):
    print("Writing suggestions to file...")
    with open('suggestions.txt', 'w') as f:
        for suggestion in suggestions:
            file_path = suggestion['file']
            suggestion_text = suggestion['suggestion']
            f.write(f"File: {file_path}\n\n{suggestion_text}\n\n\n")

def parse_code_changes(original_code, refactored_code):
    changes = []

    # Split the code into lines
    original_lines = original_code.splitlines()
    refactored_lines = refactored_code.splitlines()

    # Compare the lines using difflib
    diff = list(difflib.ndiff(original_lines, refactored_lines))

    # Iterate through the diff and identify changes
    for index, line in enumerate(diff):
        # If the line starts with a '+', it has been added
        if line.startswith('+'):
            if index > 0 and diff[index - 1].startswith('-'):
                changes.append({
                    'action': 'modify',
                    'old_line': diff[index - 1][2:],
                    'new_line': line[2:]
                })
            else:
                changes.append({'action': 'add', 'line': line[2:]})
        # If the line starts with a '-', it has been removed
        elif line.startswith('-') and (index == len(diff) - 1 or not diff[index + 1].startswith('+')):
            changes.append({'action': 'remove', 'line': line[2:]})

    return changes

def apply_code_changes(original_code, code_changes):
    code_lines = original_code.splitlines()
    new_code_lines = []
    change_index = 0

    for line in code_lines:
        if change_index < len(code_changes) and code_changes[change_index]['line'] == line:
            change = code_changes[change_index]
            action = change['action']

            if action == 'remove':
                pass
            elif action == 'add':
                new_code_lines.append(change['line'])
            elif action == 'modify':
                new_code_lines.append(change['new_line'])

            change_index += 1
        else:
            new_code_lines.append(line)

    # Add any remaining 'add' actions that may not have matched any lines in the original code
    for change in code_changes[change_index:]:
        if change['action'] == 'add':
            new_code_lines.append(change['line'])

    return '\n'.join(new_code_lines)

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)
