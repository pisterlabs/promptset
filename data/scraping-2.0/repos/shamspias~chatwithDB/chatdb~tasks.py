import openai
from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings
from .models import DatabaseConfig

from .database_helper import QueryView

logger = get_task_logger(__name__)


@shared_task
def get_bot_response(message_list, system_prompt, language, model_from, model_name, api_key, model_endpoint,
                     model_api_version, temperature, database_name):
    # Load the Pinecone index
    base_index = QueryView

    # Add extra text to the content of the last message
    last_message = message_list[-1]

    query_text = last_message["content"]

    # Get the most similar documents to the last message
    try:
        database_obj = DatabaseConfig.objects.get(name__exact=database_name)

        docs = base_index.find_similar_data(db_config=database_obj, message=query_text)
        print("_____________________________")
        print(docs)
        print("_____________________________")

        updated_content = query_text + "\n\nreference:\n"

        updated_content += docs + "\n\n"
    except Exception as e:
        logger.error(f"Failed to get similar documents: {e}")
        updated_content = query_text

    # Create a new HumanMessage object with the updated content
    # updated_message = HumanMessage(content=updated_content)
    updated_message = {"role": "user", "content": updated_content}

    # Replace the last message in message_list with the updated message
    message_list[-1] = updated_message

    if model_from == "azure":

        try:
            openai.api_type = model_from
            openai.api_base = model_endpoint
            openai.api_version = model_api_version
            openai.api_key = api_key
            gpt3_stream_response = openai.ChatCompletion.create(
                engine=model_name,
                stream=True,
                temperature=temperature,
                messages=[
                             {"role": "system",
                              "content": f"{system_prompt} {language} only."},
                         ] + message_list
            )
        except Exception as e:
            print(str(e))

            openai.api_type = "open_ai"
            openai.api_base = "https://api.openai.com/v1"
            openai.api_version = settings.OPENAI_AI_API_VERSION
            openai.api_key = settings.OPENAI_API_KEY

            gpt3_stream_response = openai.ChatCompletion.create(
                model=settings.OPENAI_AI_MODEL_NAME,
                stream=True,
                temperature=temperature,
                messages=[
                             {"role": "system",
                              "content": f"{system_prompt} {language} only."},
                         ] + message_list
            )

    else:
        openai.api_key = api_key
        gpt3_stream_response = openai.ChatCompletion.create(
            model=model_name,
            stream=True,
            temperature=temperature,
            messages=[
                         {"role": "system",
                          "content": f"{system_prompt} {language} only."},
                     ] + message_list
        )

    response_text = ""

    # Collect the streamed responses
    for i in gpt3_stream_response:
        if 'choices' in i and i['choices']:
            if 'delta' in i['choices'][0] and 'content' in i['choices'][0]['delta']:
                response_text += i['choices'][0]['delta']['content']

    # bot_message = gpt3_response["choices"][0]["message"]["content"].strip()

    return response_text
