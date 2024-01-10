import json
import os
import re
from syslog import LOG_PERROR
from fastapi import HTTPException
from openai import OpenAI, OpenAIError
from app.core.config import settings
from app.models.database import chats, threads, project_chat_association, chat_thread_association
from app.dependencies import get_database
from sqlalchemy import select
import time
import datetime
from app.utils.project_utils import get_project_folder_path_util
from typing import List, Dict
from app.utils.project_utils import (
    save_project_idea_final_util,
    save_project_technical_plan_util,
    get_project_technical_plan_util,
    get_project_idea_initial_util,
    get_project_idea_final_util
    )

client = OpenAI(api_key=settings.openai_api_key)
database = get_database()

async def create_chat_thread_util():
    try:
        return client.beta.threads.create()
    except OpenAIError as e:
        raise e

async def send_initial_message_util(thread_id, assistant_id, initial_message):
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=initial_message
        )
        return client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id
        )
    except OpenAIError as e:
        raise e

async def poll_for_completion_util(thread_id, run_id, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        if run_status.status == "completed":
            return True
        elif run_status.status in ["failed", "cancelled", "expired"]:
            return False
        time.sleep(1)
    return False

async def get_assistant_messages_util(thread_id):
    try:
        messages_response = client.beta.threads.messages.list(thread_id=thread_id)
        assistant_messages = []
        
        for msg in messages_response.data:
            if msg.role == "assistant":
                for cp in msg.content:
                    if cp.type == 'text' and hasattr(cp, 'text'):
                        assistant_messages.append(cp.text.value)

        return assistant_messages
    except OpenAIError as e:
        raise e

async def delete_project_chats_util(project_id: int):
    try:
        # Retrieve chat IDs associated with the project
        chat_ids_query = project_chat_association.select().where(
            project_chat_association.c.project_id == project_id
        )
        chat_ids = await database.fetch_all(chat_ids_query)

        # Retrieve thread associations associated with each chat
        for chat in chat_ids:
            thread_associations_query = chat_thread_association.select().where(
                chat_thread_association.c.chat_id == chat['chat_id']
            )
            thread_associations = await database.fetch_all(thread_associations_query)

            # Delete each thread from OpenAI and database using the OpenAI thread_id
            for association in thread_associations:
                # Retrieve the actual OpenAI thread_id from the threads table
                thread_query = threads.select().where(
                    threads.c.id == association['thread_id']
                )
                thread = await database.fetch_one(thread_query)

                if thread:
                    # Use the OpenAI API to delete the thread
                    client.beta.threads.delete(thread_id=thread['thread_id'])

                    # Delete thread from database
                    delete_thread_query = threads.delete().where(
                        threads.c.id == association['thread_id']
                    )
                    await database.execute(delete_thread_query)

            # Delete the associations from database
            delete_thread_association_query = chat_thread_association.delete().where(
                chat_thread_association.c.chat_id == chat['chat_id']
            )
            await database.execute(delete_thread_association_query)

        # Delete chat associations from the database
        delete_chat_association_query = project_chat_association.delete().where(
            project_chat_association.c.project_id == project_id
        )
        await database.execute(delete_chat_association_query)

        # Delete chats from the database
        delete_chats_query = chats.delete().where(
            chats.c.id.in_([chat['chat_id'] for chat in chat_ids])
        )
        await database.execute(delete_chats_query)

        return {"message": f"All chats and associated threads for project {project_id} have been deleted"}
    except Exception as e:
        raise e

async def delete_specific_chat_and_threads(chat_id: int):
    try:
        # Retrieve thread associations with the specific chat
        thread_associations_query = chat_thread_association.select().where(
            chat_thread_association.c.chat_id == chat_id
        )
        thread_associations = await database.fetch_all(thread_associations_query)

        # Delete each thread from OpenAI and database using the OpenAI thread_id
        for association in thread_associations:
            # Retrieve the OpenAI thread_id from the threads table
            thread_query = threads.select().where(
                threads.c.id == association['thread_id']
            )
            thread = await database.fetch_one(thread_query)

            if thread:
                # Use the OpenAI API to delete the thread
                client.beta.threads.delete(thread_id=thread['thread_id'])

                # Delete thread from database
                delete_thread_query = threads.delete().where(
                    threads.c.id == association['thread_id']
                )
                await database.execute(delete_thread_query)

        # Delete the chat threads associations from database
        delete_association_query = chat_thread_association.delete().where(
            chat_thread_association.c.chat_id == chat_id
        )
        await database.execute(delete_association_query)

        # Delete project chat associations from the database
        delete_chat_association_query = project_chat_association.delete().where(
            project_chat_association.c.chat_id == chat_id
        )
        await database.execute(delete_chat_association_query)

        # Delete the specific chat from the database
        delete_chat_query = chats.delete().where(
            chats.c.id == chat_id
        )
        await database.execute(delete_chat_query)

        return {"message": f"Chat with ID {chat_id} and associated threads have been deleted"}
    except Exception as e:
        raise e

async def save_conversation_util(chat_id, conversation):
    try:
        update_query = chats.update().where(
            chats.c.id == chat_id
        ).values(
            chat_messages=json.dumps(conversation)
        )
        await database.execute(update_query)
    except Exception as e:
        raise e

async def fetch_conversation_util(chat_id: int):
    try:
        query = select([chats.c.chat_messages]).where(chats.c.id == chat_id)
        result = await database.fetch_one(query)
        if result:
            return json.loads(result['chat_messages'])
        else:
            return None
    except Exception as e:
        raise e

async def get_project_openai_chat_thread_ids_util(project_id: int):
    # Define a SQL query to join project_chat_association with chats
    # to fetch the OpenAI chat IDs (chat_id from chats table)
    join_query = select([
        chats.c.chat_thread_id
    ]).select_from(
        project_chat_association.join(chats, project_chat_association.c.chat_id == chats.c.id)
    ).where(
        project_chat_association.c.project_id == project_id
    )

    # Execute the query and fetch results
    result = await database.fetch_all(join_query)
    return [row['chat_thread_id'] for row in result]

async def chat_thread_exists_util(chat_name: str, primary_to_secondary_name: str, secondary_to_primary_name: str):
    # Check if the chat exists
    chat_query = chats.select().where(
        chats.c.chat_name == chat_name
    )
    chat_result = await database.fetch_one(chat_query)

    if chat_result:
        return True, chat_result['id']  # Return True and the found chat_id

    # Check if the threads exist
    thread_query = threads.select().where(
        (threads.c.thread_name == primary_to_secondary_name) | 
        (threads.c.thread_name == secondary_to_primary_name)
    )
    thread_exists = await database.fetch_one(thread_query) is not None

    return thread_exists, None  # Return False and None if no chat_id is found

async def insert_chat_data_util(chat_name, chat_assistant_primary, chat_assistant_secondary, chat_messages):
    chat_query = chats.insert().values(
        chat_name=chat_name,
        chat_assistant_primary=chat_assistant_primary,
        chat_assistant_secondary=chat_assistant_secondary,
        chat_messages=json.dumps(chat_messages),
        created_at=datetime.datetime.now()
    )
    chat_id = await database.execute(chat_query)
    return chat_id

async def insert_thread_data_util(thread_id, thread_name):
    thread_query = threads.insert().values(
        thread_id=thread_id,
        thread_name=thread_name,
        created_at=datetime.datetime.now()
    )
    thread_db_id = await database.execute(thread_query)
    return thread_db_id

async def associate_chat_with_project_util(project_id, chat_id):
    association_query = project_chat_association.insert().values(
        project_id=project_id,
        chat_id=chat_id
    )
    await database.execute(association_query)

async def associate_thread_with_chat_util(chat_id, thread_id):
    association_query = chat_thread_association.insert().values(
        chat_id=chat_id,
        thread_id=thread_id
    )
    await database.execute(association_query)

async def get_source_code_from_folder_util(folder_path):
    file_contents = {}
    # List of file extensions to consider
    code_extensions = ['.py', '.html', '.css', '.js', '.txt'] # Add or remove extensions as needed

    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in code_extensions):
            with open(os.path.join(folder_path, filename), 'r') as file:
                file_contents[filename] = file.read()

    return file_contents

async def get_all_source_codes_from_folder_util(folder_path):
    file_contents = {}
    # Dictionary mapping file extensions to code types
    code_extensions = {
        '.py': 'python',
        '.html': 'html',
        '.css': 'css',
        '.js': 'javascript',
    }

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in code_extensions:
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as file:
                    # Wrap the file content with code fences
                    code_type = code_extensions[ext]
                    file_contents[filename] = f"```{code_type}\n{file.read()}\n```"

    return file_contents

async def format_initial_message(chat_type, template, id, tech_scope, chat_goal, max_exchanges, chat_end, response_from_secondary_assistant, source_code_folder=None):
    try:
        idea_initial, idea_final, technical_plan, source_code = None, None, None, None

        if chat_type in ["stakeholder_consultant"]:
            idea_initial = await get_project_idea_initial_util(id)
        if chat_type in ["ceo_cto"]:
            idea_final = await get_project_idea_final_util(id)
        if chat_type in ["cto_programmer", "programmer_tester", "cto_technical-writer", "ceo_user-documentation"]:
            technical_plan = await get_project_technical_plan_util(id)
        if chat_type in ["programmer_tester"] and source_code_folder:
            source_code = await get_source_code_from_folder_util(source_code_folder)
        if chat_type in ["cto_technical-writer", "ceo_user-documentation"] and source_code_folder:
            source_code = await get_all_source_codes_from_folder_util(source_code_folder)

        return template.format(
            tech_scope=tech_scope,
            chat_goal=chat_goal,
            idea_initial=idea_initial,
            idea_final=idea_final,
            technical_plan=technical_plan,
            source_code=source_code,
            max_exchanges=max_exchanges,
            chat_end=chat_end,
            response_from_secondary=response_from_secondary_assistant
        )
    except Exception as e:
        # Log the error for debugging
        LOG_PERROR("Error in formatting message: " + str(e))
        return None

async def request_and_process_final_output(project_id,
                                           chat_type,
                                           primary_secondary_chat_thread_data,
                                           secondary_primary_chat_thread_data,
                                           secondary_assistant_id,
                                           output_format_instructions,
                                           output_request,
                                           conversation,
                                           sender_name_primary,
                                           sender_name_secondary
                                           ):
    # Send a new message to the secondary assistant asking for the final output
    final_output_request = output_format_instructions + " - " + output_request
    
    # Append Primary Assistant response to the conversation
    conversation.append({"sender": sender_name_primary, "message": final_output_request})
    
    request_to_secondary_for_output = await send_initial_message_util(
        thread_id=primary_secondary_chat_thread_data.id,
        assistant_id=secondary_assistant_id,
        initial_message=final_output_request
    )

    # Wait for the response
    if not await poll_for_completion_util(primary_secondary_chat_thread_data.id, request_to_secondary_for_output.id):
        raise HTTPException(status_code=500, detail="Error: In requesting final output")

    final_output_messages = await get_assistant_messages_util(primary_secondary_chat_thread_data.id)
    final_output_from_secondary = final_output_messages[0]

    # Append Secondary Assistant response to the conversation
    conversation.append({"sender": sender_name_secondary, "message": final_output_from_secondary})
    
    # Directly use the response as the final output
    output_content = final_output_from_secondary.strip()

    # Save the output based on the chat type
    if chat_type == "stakeholder_consultant":
        await save_project_idea_final_util(project_id=project_id, final_idea=output_content)
    elif chat_type == "ceo_cto":
        await save_project_technical_plan_util(project_id=project_id, technical_plan=output_content)
    elif chat_type == "cto_programmer":
        await save_code_to_file_util(project_id=project_id, thread_id_primary=primary_secondary_chat_thread_data.id, thread_id_secondary=secondary_primary_chat_thread_data.id)
    elif chat_type == "programmer_tester":
        await save_code_to_file_util(project_id=project_id, thread_id_primary=primary_secondary_chat_thread_data.id, thread_id_secondary=secondary_primary_chat_thread_data.id, custom_file_name="test_app.py")
    elif chat_type == "cto_technical-writer":
        await save_code_to_file_util(project_id=project_id, thread_id_primary=primary_secondary_chat_thread_data.id, thread_id_secondary=secondary_primary_chat_thread_data.id, process_all_messages=False, include_markdown=True, custom_file_name="technical_document.md")
    elif chat_type == "ceo_user-documentation":
        await save_code_to_file_util(project_id=project_id, thread_id_primary=primary_secondary_chat_thread_data.id, thread_id_secondary=secondary_primary_chat_thread_data.id, process_all_messages=False, include_markdown=True, custom_file_name="user_manual_document.md")

    return True

async def save_and_record_conversation(chat_id, project_id, chat_name, conversation):
    """
    Save the conversation in the database and to a JSON file.
    
    Args:
        chat_id (int): The ID of the chat.
        project_id (int): The ID of the project.
        chat_name (str): The name of the chat.
        conversation (list): The conversation data to be saved.

    Raises:
        HTTPException: If an error occurs in saving the conversation.
    """
    # Save the conversation in the database
    await save_conversation_util(chat_id=chat_id, conversation=conversation)
    
    # Save the conversation to a JSON file
    success = await save_conversation_to_file_util(project_id=project_id, chat_name=chat_name, conversation=conversation)
    if not success:
        raise HTTPException(status_code=500, detail="Error: Saving conversation to file")

async def initialize_chat_and_threads(id, request_body):
    """
    Initialize chat and threads for the conversation.
    
    Args:
        id (int): The ID of the project.
        request_body (CreateChatRequest): The request body containing chat details.

    Returns:
        tuple: A tuple containing the chat ID and the thread data for primary and secondary assistants.

    Raises:
        HTTPException: If an error occurs during the initialization.
    """
    primary_to_secondary_name = request_body.chat_name + "-primary-to-secondary"
    secondary_to_primary_name = request_body.chat_name + "-secondary-to-primary"

    # Check if the chat or its threads already exist
    exists, chat_id = await chat_thread_exists_util(
        chat_name=request_body.chat_name,
        primary_to_secondary_name=primary_to_secondary_name,
        secondary_to_primary_name=secondary_to_primary_name
    )
    if exists:
        return chat_id, None, None 

    # Create chat threads
    primary_secondary_chat_thread_data = await create_chat_thread_util()
    secondary_primary_chat_thread_data = await create_chat_thread_util()

    # Insert chat data
    chat_id = await insert_chat_data_util(
        chat_name=request_body.chat_name,
        chat_assistant_primary=request_body.chat_assistant_primary,
        chat_assistant_secondary=request_body.chat_assistant_secondary,
        chat_messages=""
    )

    # Insert thread data
    primary_secondary_thread_id = await insert_thread_data_util(
        thread_id=primary_secondary_chat_thread_data.id,
        thread_name=primary_to_secondary_name
    )
    secondary_primary_thread_id = await insert_thread_data_util(
        thread_id=secondary_primary_chat_thread_data.id,
        thread_name=secondary_to_primary_name
    )

    # Associate the chat with the project
    await associate_chat_with_project_util(id, chat_id)

    # Associate the thread with the chat
    await associate_thread_with_chat_util(chat_id, primary_secondary_thread_id)
    await associate_thread_with_chat_util(chat_id, secondary_primary_thread_id)

    return chat_id, primary_secondary_chat_thread_data, secondary_primary_chat_thread_data

async def list_message_files(thread_id, message_id):
    try:
        response = client.beta.threads.messages.files.list(thread_id=thread_id, message_id=message_id)
        return response
    except OpenAIError as e:
        raise e

async def list_thread_messages(thread_id):
    try:
        return client.beta.threads.messages.list(thread_id=thread_id)
    except OpenAIError as e:
        raise e

async def save_conversation_to_file_util(project_id: int, chat_name: str, conversation: List[Dict]):
    try:
        # Get the project folder path
        folder_path = await get_project_folder_path_util(project_id)

        # Create the 'logs' directory if it does not exist
        logs_folder_path = os.path.join(folder_path, "logs")
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
        
        # Format the file name
        json_name = chat_name.replace('-', '_').lower()
        file_name = f"logs_chat_messages_{json_name}.json"
        
        # Define the complete file path
        conversation_file_path = os.path.join(logs_folder_path, file_name)

        # Save the conversation to a JSON file
        with open(conversation_file_path, 'w') as file:
            json.dump(conversation, file, indent=4)
        
        return True
    except Exception as e:
        # Handle any exceptions (logging, custom error handling, etc.)
        print(f"Error saving conversation to file: {e}")
        return False

async def save_code_to_file_util(project_id: int, thread_id_primary: str, thread_id_secondary: str, process_all_messages: bool = True, include_markdown: bool = False, custom_file_name: str = None):
    try:
        files_found = False

        folder_path = await get_project_folder_path_util(project_id)

        templates_path = os.path.join(folder_path, "templates")
        static_path = os.path.join(folder_path, "static")
        documentation_path = os.path.join(folder_path, "documentation")
        os.makedirs(templates_path, exist_ok=True)
        os.makedirs(static_path, exist_ok=True)
        os.makedirs(documentation_path, exist_ok=True)

        # Function to handle the retrieval and saving of files
        async def handle_files_in_message(message):
            nonlocal files_found

            if message.file_ids:
                files_found = True
                for file_id in message.file_ids:
                    file_metadata = client.files.retrieve(file_id=file_id)
                    
                    file_response = client.files.content(file_id=file_id)

                    # Extract just the filename from the full path
                    filename_only = os.path.basename(file_metadata.filename)
                    file_extension = os.path.splitext(filename_only)[1]  # Extract extension from filename

                    # Determine correct directory based on file type
                    if file_extension in ['.css', '.js']:
                        file_path = os.path.join(static_path, filename_only)
                    elif file_extension in ['.html']:
                        file_path = os.path.join(templates_path, filename_only)
                    elif file_extension in ['.md']:
                        file_path = os.path.join(documentation_path, filename_only)
                    else:
                        file_path = os.path.join(folder_path, filename_only)

                    with open(file_path, 'wb') as file:  # Open in binary write mode
                        file.write(file_response.content)  # Write the binary content

        # Process messages for both primary and secondary threads
        for thread_id in [thread_id_primary, thread_id_secondary]:
            messages_response = await list_thread_messages(thread_id)

            for message in messages_response.data:
                await handle_files_in_message(message)

        # Fallback mechanism: parse messages for code if no files were found
        if not files_found:
            output_content = ""
            messages_response_for_parsing = await list_thread_messages(thread_id_primary)
            if process_all_messages:
                output_content = " ".join([cp.text.value for msg in messages_response_for_parsing.data for cp in msg.content if cp.type == 'text' and hasattr(cp, 'text')])
            elif messages_response_for_parsing.data:
                last_message = messages_response_for_parsing.data[0]
                output_content = " ".join([cp.text.value for cp in last_message.content if cp.type == 'text' and hasattr(cp, 'text')])

            await parse_and_save_code_snippets(output_content, folder_path, templates_path, static_path, documentation_path, custom_file_name, include_markdown)

        return True

    except Exception as e:
        print(f"Error saving code to file: {e}")
        return False

async def parse_and_save_code_snippets(output_content, folder_path, templates_path, static_path, documentation_path, custom_file_name: str = None, include_markdown: bool = False):
    try:
        # Define code type patterns and corresponding filenames
        code_patterns = {
            'py': (r"```python\s+(.*?)\s+```", custom_file_name if custom_file_name else "app.py", folder_path),
            'html': (r"```html\s+(.*?)\s+```", "index.html", templates_path),
            'css': (r"```css\s+(.*?)\s+```", "style.css", static_path),
            'js': (r"```js\s+(.*?)\s+```", "script.js", static_path),
            # Add a placeholder pattern for Markdown if it's included
            'md': (None, custom_file_name if custom_file_name else "documentation.md", documentation_path) if include_markdown else None,
        }

        # Remove any None entries from code_patterns
        code_patterns = {k: v for k, v in code_patterns.items() if v is not None}

        # Save Markdown content only if include_markdown is True and output_content is not empty
        if include_markdown and output_content.strip():
            md_filename = custom_file_name if custom_file_name else "documentation.md"
            md_file_path = os.path.join(documentation_path, md_filename)
            with open(md_file_path, 'w') as file:
                file.write(output_content)
        
        # Iterate through patterns and save files with fixed names
        for pattern, filename, target_path in code_patterns.values():
            # Handle non-regex patterns (like Markdown)
            if pattern is None:
                continue

            matches = re.finditer(pattern, output_content, re.DOTALL)
            for match in matches:
                code_content = match.group(1)

                # Define the complete file path
                file_path = os.path.join(target_path, filename)

                # Save the code content to a file
                with open(file_path, 'w') as file:
                    file.write(code_content)

    except Exception as e:
        print(f"Error saving code to file: {e}")
        return False
    return True