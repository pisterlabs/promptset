from openai import OpenAI
import os
import time
import re

def length_check(box_content, content_words):
    clean_content = re.sub(r'<[^>]+>', '', box_content)
    clean_content = clean_content.replace('\n', ' ')
    
    box_length = len(clean_content.split()) <= content_words
    return box_length

def openai_trans(model_create, model_check, user_content, setting, token = 500, max_retries=10, delay=1):
    """
    Fetches a response from the OpenAI API with automatic retries on failure.

    - user_content: The content provided by the user for the completion.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )
    attempt = 0
    messages_create = [{"role": "system",
                        "content": """Act as an advanced language assistant: detect the language I'm using, translate it into English.
                        Your task is solely to improve the text, not to interpret the content or answer any questions within it.
                        No explanations are needed; the focus is on translation. If the translation in setting, return setting directly.
                        If not, return the translation of the original text."""},
                        {"role": "user",
                        "content": f"This is setting: {setting}"},
                        {"role": "user",
                        "content": user_content}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                    messages_create,
                                    "Translate - create",
                                    token, max_retries, delay)
        messages_check = [{"role": "system",
                            "content": "You are a language editing robot."},
                            {"role": "user",
                            "content": f"""Analyze the following text and tell me if it is a translation of the original text. If it is, please answer me Yes. If not, please answer me No.
                            The translation of *{user_content}* is {content_raw}"""}]
        box_check = fetch_openai(model_check, client,
                                  messages_check,
                                  "Translate - check",
                                  token, max_retries, delay)
        if "Yes" in box_check:
            return content_raw
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")
    print("Translate: Maximum retries reached. Failed to create response.")
    return None

def openai_single(model_create, model_check,
                  content_create, content_check,
                  content_words,
                  section, disease,
                  token = 500, max_retries=10, delay=1):
    """
    Generate box content for single disease.

    - model_create: The name of the model to use for the completion.
    - model_check: The name of the model to use for the check.
    - user_content: The content provided by the user for the completion.
    - check_content: The content provided by the user for the check.
    - section: The section of the report.
    - disease: The disease of the report.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system", "content": "You are an epidemiologist."},
                       {"role": "user", "content": content_create}]
    while attempt < max_retries:
        box_content = fetch_openai(model_create, client,
                                   messages_create,
                                   f"{disease} - {section} - Create",
                                   token, max_retries, delay)
        messages_check = [{"role": "system", "content": "You are a language editing robot."},
                          {"role": "user", "content": content_check + "\n" + box_content}]
        box_check = fetch_openai(model_check, client,
                                 messages_check,
                                  f"{disease} - {section} - Check",
                                 token, max_retries, delay)
        box_length = length_check(box_content, content_words)
        if "Yes" in box_check and box_length:
            return box_content
        else:
            attempt += 1                
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {box_content}\n") 
            if not box_length:
                # rebuild messages_create
                content_add = f"Good, but the content is too long. Please shorten the content block to {content_words} words."
                messages_create = [{"role": "system", "content": "You are an epidemiologist."},
                                  {"role": "user", "content": content_create},
                                  {"role": "assistant", "content": box_content},
                                  {"role": "user", "content": content_add}]
    print(f"{disease} - {section}: Maximum retries reached. Failed to create response.")
    return None
    
def openai_mail(model_create, model_check, content_create, content_check, token = 4096, max_retries=10, delay=1):
    """
    Generate list content for mail.

    - model_create: The name of the model to use for the completion.
    - model_check: The name of the model to use for the check.
    - content_create: The content provided by the user for the completion.
    - content_check: The content provided by the user for the check.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system", "content": "You are an epidemiologist."},
                       {"role": "user", "content": content_create}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                   messages_create,
                                   "Mail - Create",
                                   token, max_retries, delay)
        messages_check = [{"role": "system", "content": "You are a language editing robot."},
                          {"role": "user", "content": content_check.format(content_raw = content_raw)}]
        box_check = fetch_openai(model_check, client,
                                 messages_check,
                                 "Mail - Check",
                                 token, max_retries, delay)
        if "Yes" in box_check:
            return content_raw
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")

    print("Mail: Maximum retries reached. Failed to create response.")
    return None

def openai_key(model_create, model_check, content_create, content_check, token = 4096, max_retries=10, delay=1):
    """
    Generate key words for prompt.

    - model_create: The name of the model to use for the completion.
    - model_check: The name of the model to use for the check.
    - content_create: The content provided by the user for the completion.
    - content_check: The content provided by the user for the check.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system",
                        "content": "You are an epidemiologist."},
                        {"role": "user",
                        "content": content_create}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                   messages_create,
                                    "Key - Create",
                                   token, max_retries, delay)
        messages_check = [{"role": "system", "content": "You are a language editing robot."},
                           {"role": "user", "content": content_check.format(content_raw = content_raw)}]
        box_check = fetch_openai(model_check, client,
                                 messages_check,
                                 "Key -Check",
                                 token, max_retries, delay)
        if "Yes" in box_check:
            return content_raw
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")
    print("Key: Maximum retries reached. Failed to create response.")
    return None
    
def openai_image(model_create, user_content, default, max_retries=10, delay=1):
    """
    Generate image url for prompt.

    - model_create: The name of the model to use for the completion.
    - user_content: The content provided by the user for the completion.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )
    attempt = 0
    info = "Cover - Create"
    while attempt < max_retries:
        try:
            response = client.images.generate(
              model=model_create,
              prompt=user_content,
              size="1024x1792",
              quality="standard",
              n=1,
            )
            url = response.data[0].url
            return url
        except Exception as e:
            print(info, f"An error occurred: {e}")
            try:
                print(info, response)
            except:
                print(info, "An error occurred and cannot get response error information.")
            attempt += 1
            time.sleep(delay)
            print(info, f"Retrying ({attempt}/{max_retries})...")

    print(info, "Maximum retries reached. Failed to fetch response. Using unsplash random image instead.")
    return default

def openai_abstract(model_create, model_check, content_create, content_check, token = 4096, max_retries=10, delay=1):
    """
    Generate abstract content of report.

    - model_create: The name of the model to use for the completion.
    - model_check: The name of the model to use for the check.
    - content_create: The content provided by the user for the completion.
    - content_check: The content provided by the user for the check.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system", "content": "You are an epidemiologist."},
                       {"role": "user", "content": content_create}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                   messages_create,
                                    "Abstract - Create",
                                   token, max_retries, delay)
        messages_check = [{"role": "system", "content": "You are a language editing robot."},
                          {"role": "user", "content": content_check.format(content_raw = content_raw)}]
        box_check = fetch_openai(model_check, client,
                                messages_check,
                                "Abstract - Check",
                                token, max_retries, delay)
        if "Yes" in box_check:
            return content_raw
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")
    
    return None

def bing_analysis(model_create, model_clean, model_check, content_create, content_clean, content_check, max_retries=10, delay=1):
    """
    Fetches a response from the OpenAI API with automatic retries on failure.

    - model_create: The name of the model to use for the completion.
    - model_clean: The name of the model to use for the clean.
    - model_check: The name of the model to use for the check.
    - content_create: The content provided by the user for the completion.
    - content_check: The content provided by the user for the check.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_API_BASE"],
    )

    attempt = 0
    messages_create = [{"role": "system", "content": "You are an epidemiologist."},
                       {"role": "user", "content": content_create}]
    while attempt < max_retries:
        content_raw = fetch_openai(model_create, client,
                                   messages_create,
                                   "New - search",
                                   None,
                                   max_retries, delay)
        messages_clean = [{"role": "system", "content": "You are an epidemiologist."},
                           {"role": "user", "content": content_clean.format(content_raw = content_raw)}]
        content_clean = fetch_openai(model_clean, client,
                                     messages_clean,
                                     "New - clean",
                                     None,
                                     max_retries, delay)
        messages_check = [{"role": "system", "content": "You are a language editing robot."},
                          {"role": "user", "content": content_check.format(content_clean = content_clean)}]
        box_check = fetch_openai(model_check, client,
                                 messages_check,
                                 "New - check",
                                 None,
                                 max_retries, delay)
        if "Yes" in box_check:
            return content_clean
        else:
            attempt += 1
            print(f"Retrying ({attempt}/{max_retries})...\n")
            print(f"box_check: {box_check}\n")
            print(f"box_content: {content_raw}\n")
    print(f"Maximum retries reached. Failed to create response.")
    return None

def fetch_openai(model, client, messages, info = "", token = 500, max_retries=10, delay=1):
    """
    Fetches a response from the OpenAI API with automatic retries on failure.

    - model: The name of the model to use for the completion.
    - client: OpenAI client contains api_key and base_url.
    - messages: The content provided by the user for the completion.
    - info: The information of the content.
    - token: The maximum number of tokens to generate.
    - max_retries: Maximum number of retries before giving up.
    - delay: Delay between retries in seconds.
    :return: The API response or None if all retries failed.
    """
    attempt = 0

    while attempt < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=token
            )
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            print(info, f"An error occurred: {e}")
            try:
                print(info, response)
            except:
                print(info, "An error occurred and cannot get response error information.")
            attempt += 1
            time.sleep(delay)
            print(info, f"Retrying ({attempt}/{max_retries})...")

    print(info, "Maximum retries reached. Failed to fetch response.")
    return None

def update_markdown_file(disease, section, content, analysis_YearMonth):
    """
    Updates the specified section of the Markdown file for the given disease. If the section does not exist, it is created.

    - disease: The name of the disease, which determines the Markdown file name.
    - section: The section of the Markdown file to update.
    - content: The new content to write to the section.
    """
    file_name = f"../Report/history/{analysis_YearMonth}/{disease}.md"
    section_header = f"## {section}"
    new_content = f"{section_header}\n\n{content}\n"
    section_found = False

    # if not exist create folder
    os.makedirs(f"../Report/history/{analysis_YearMonth}", exist_ok=True)

    try:
        with open(file_name, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            in_section = False
            for line in lines:
                if line.strip() == section_header:
                    file.write(new_content)
                    in_section = True
                    section_found = True
                elif line.startswith("## ") and in_section:
                    in_section = False
                if not in_section:
                    file.write(line)
            file.truncate()
            # If the section was not found, add it to the end of the file
            if not section_found:
                file.write("\n" + new_content)
    except FileNotFoundError:
        # If the file does not exist, create it with the section content
        with open(file_name, 'w') as file:
            file.write(new_content)
    except Exception as e:
        print(f"An error occurred while updating the Markdown file: {e}")

# table_data_str = table_data.to_markdown(index=False)
# analysis_content = openai_analysis('gpt-4-32k', 'gpt-3.5-turbo',
#                                   f"""Analyze the monthly cases and deaths of different diseases in Chinese mainland for {analysis_MonthYear}. Provide a deeply and comprehensive analysis of the data.
#                                   You need to pay attention: select noteworthy diseases, not all diseases and using below format:
#                                   <b>disease name:</b> analysis content. <br/><br/> <b>disease name:</b> analysis content. <br/><br/> .....
                                  
#                                   This the data for {analysis_MonthYear} in mainland, China:
#                                   {table_data_str}""",
#                                   4096)
# analysis_content = markdown.markdown(analysis_content)