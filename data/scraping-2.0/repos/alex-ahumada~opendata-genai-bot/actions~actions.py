# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

import os
import io
import re
import urllib.parse
import datetime
import requests
import json
import pandas as pd
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage


# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# from .helpers.openai_info import get_openai_callback
from .helpers.openai import get_openai_token_cost_for_model
import boto3
from botocore.exceptions import ClientError
from typing import Any, Optional, Text, Dict, List
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import seaborn as sns

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from rasa_sdk import Action, FormValidationAction, Tracker, logger
from rasa_sdk.events import SlotSet, EventType, AllSlotsReset, UserUtteranceReverted
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

load_dotenv()

ALLOWED_FILE_FORMATS = ["csv", "xls", "xlsx", "pdf"]

aggregate_titles = ["total", "totales"]

# Create a new client and connect to the server
mongo_client = MongoClient(os.getenv("MONGODB_URI"), server_api=ServerApi("1"))

# Set OpenAI API key
client = OpenAI()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# llm = OpenAI(api_token=os.getenv("OPENAI_API_KEY"))


def get_completion(
    conversation_id: str,
    rasa_action: str,
    prompt: str,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4"),
) -> str:
    response: ChatCompletion = client.chat.completions.create(
        model=model,  # this is the model that the API will use to generate the response
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You are answering questions about a dataset. The user is located in Spain, so you should answer in Spanish and use the metric system, currencies will be in euros. Please avoid answering questions not related with the dataset or that are offensive or unehtical",
            },
            {"role": "user", "content": prompt},
        ],  # this is the prompt that the model will complete
        temperature=0.5,  # this is the degree of randomness of the model's output
        max_tokens=int(
            os.environ.get("OPENAI_MAX_TOKENS", 2000)
        ),  # this is the maximum number of tokens that the model can generate
        top_p=1,  # this is the probability that the model will generate a token that is in the top p tokens
        frequency_penalty=0,  # this is the degree to which the model will avoid repeating the same line
        presence_penalty=0,  # this is the degree to which the model will avoid generating offensive language
    )

    # Log completion to MongoDB
    try:
        mongo_client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
        db = mongo_client.get_database("logs")
        collection = db.get_collection("completions")
        document = {
            "created": response.created,
            "conversation_id": conversation_id,
            "model": response.model,
            "rasa_action": rasa_action,
            "prompt": prompt,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": get_openai_token_cost_for_model(
                    model_name=model, num_tokens=response.usage.total_tokens
                ),
            },
        }
        collection.insert_one(document)
    except Exception as e:
        print(e)

    return response.choices[0].message.content


# def get_completion_with_pandasai(
#     conversation_id: str,
#     rasa_action: str,
#     prompt: str,
#     dataframe: pd.DataFrame,
#     model: str = os.environ.get("OPENAI_MODEL", "gpt-4"),
# ) -> str:
#     sdf = SmartDataframe(
#         df=dataframe,
#         config={
#             "llm": llm,
#             "custom_instructions": "The query will be made in Spanish and the results will be returned in Spanish.",
#         },
#     )
#     sdf.chat(prompt)
#     with get_openai_callback() as cb:
#         response = sdf.chat(prompt)
#         print(response)
#         print(cb)

#     # Log completion to MongoDB
#     try:
#         client.admin.command("ping")
#         print("Pinged your deployment. You successfully connected to MongoDB!")
#         db = client.get_database("logs")
#         collection = db.get_collection("completions")
#         document = {
#             "created": datetime.datetime.now(),
#             "conversation_id": conversation_id,
#             "model": model,
#             "rasa_action": rasa_action,
#             "prompt": prompt,
#             "usage": {
#                 "prompt_tokens": cb.prompt_tokens,
#                 "completion_tokens": cb.completion_tokens,
#                 "total_tokens": cb.total_tokens,
#                 "cost": cb.total_cost,
#             },
#         }
#         collection.insert_one(document)
#     except Exception as e:
#         print(e)

#     return response


def create_menu(conversation_id, message, buttons):
    keyboard = json.dumps({"inline_keyboard": buttons})

    request_url = (
        f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/sendMessage"
    )
    params = {
        "chat_id": conversation_id,
        "text": message,
        "reply_markup": keyboard,
    }

    try:
        response = requests.post(
            request_url,
            params=params,
        )
    except Exception as e:
        print(e)

    return response


def clear_menu(
    conversation_id: str,
    message_id: str,
    dispatcher: CollectingDispatcher,
    action_name: str = "",
):
    # Define the Telegram API URL and the parameters for the deleteMessage method
    url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/deleteMessage"
    params = {"chat_id": conversation_id, "message_id": message_id}

    # Send the request to the Telegram API
    response = requests.post(url, params=params)

    # Check the response
    if response.status_code != 200:
        dispatcher.utter_message(text=f"Failed to remove menu {action_name}")


class ValidateDataSearchTermsForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_data_search_terms_form"

    # @staticmethod
    # def required_slots(tracker: Tracker) -> List[Text]:
    #     if tracker.latest_message["intent"].get("name") == "start_over":
    #         return []
    #     else:
    #         return ["data_search_terms"]

    # async def request_next_slot(
    #     self,
    #     dispatcher: "CollectingDispatcher",
    #     tracker: "Tracker",
    #     domain: Dict[Text, Any],
    # ) -> Optional[List[EventType]]:
    #     """Request the next slot and utter template if needed,
    #     else return None"""

    #     for slot in self.required_slots(tracker):
    #         if self._should_request_slot(tracker, slot):
    #             logger.debug(f"Request next slot '{slot}'")
    #             dispatcher.utter_message(template=f"utter_ask_{slot}", **tracker.slots)
    #             return [SlotSet("requested_slot", slot)]

    #     # no more required slots to fill
    #     return None

    def validate_data_search_terms(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `data_search_terms` value."""

        url_search = f"{os.getenv('DKAN_API')}/search?fulltext={urllib.parse.quote(slot_value.lower())}"
        response_search = requests.request("GET", url_search, headers={}, data={})
        response_search_json = response_search.json()
        num_items = response_search_json["total"]

        print("url_search in validation:", url_search)

        if num_items == "0":
            dispatcher.utter_message(text="Lo siento, no tengo datos sobre ese tema.")
            return {"data_search_terms": None}
        # dispatcher.utter_message(text=f"OK! Quieres datos sobre {slot_value}.")
        return {"data_search_terms": slot_value}


class ValidateDataSelectDatasetForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_data_select_dataset_form"

    def validate_dataset_uuid(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        """Validate `dataset_uuid` value."""

        # validation logic for dataset_uuid slot
        if slot_value is not None and self._is_valid_uuid(slot_value):
            # if the dataset_uuid slot is filled and valid
            return {"dataset_uuid": slot_value}
        else:
            # if the dataset_uuid slot is not filled or invalid
            dispatcher.utter_message(
                text="El UUID no se corresponde con ningún conjunto de datos."
            )
            return {"dataset_uuid": None}

    def _is_valid_uuid(self, uuid: Text) -> bool:
        # check if the uuid is valid
        regex = r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
        return re.fullmatch(regex, uuid) is not None


class ValidateDataDownloadForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_data_download_form"

    def validate_data_file_format(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate `data_file_format` value."""

        data_meta = tracker.get_slot("data_meta")

        allowed_file_formats_dynamic = []

        for distribution in data_meta["distribution"]:
            allowed_file_formats_dynamic.append(distribution["format"].lower())

        # Check and normalize file format
        if slot_value.replace(".", "").lower() not in allowed_file_formats_dynamic:
            dispatcher.utter_message(text="El formato no es valido.")
            return {"data_file_format": None}

        return {"data_file_format": slot_value.replace(".", "").lower()}


class AskForDatasetUUID(Action):
    def name(self) -> Text:
        return "action_ask_dataset_uuid"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")
        search_terms = tracker.get_slot("data_search_terms")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        # Search dataset with search terms
        url_search = f"{os.getenv('DKAN_API')}/search?fulltext={urllib.parse.quote(search_terms)}"
        payload = {}
        headers = {}
        response_search = requests.request(
            "GET", url_search, headers=headers, data=payload
        )
        response_search_json = response_search.json()

        dataset_keys = list(response_search_json["results"].keys())

        # Generate buttons for the first 5 datasets
        buttons = []

        buttons_index_emoji = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

        for index, key in enumerate(dataset_keys[:5]):
            buttons.append(
                [
                    {
                        "text": f"{buttons_index_emoji[index]} {response_search_json['results'][key]['title']}",
                        "callback_data": response_search_json["results"][key][
                            "identifier"
                        ].lower(),
                    }
                ]
            )

        response = create_menu(
            conversation_id, "Por favor, elige un conjunto de datos.", buttons
        )

        if response.ok:
            # print("File format menu rendered.")
            # dispatcher.utter_message(text="Mostrando el menu (file format).")
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            # print("Error rendering menu (file format).")
            dispatcher.utter_message(text="Error al mostrar el menu (dataset uuid).")
            return []


class AskForDataFileFormat(Action):
    def name(self) -> Text:
        return "action_ask_data_file_format"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        data_meta = tracker.get_slot("data_meta")

        buttons = []

        for distribution in data_meta["distribution"]:
            buttons.append(
                [
                    {
                        "text": distribution["format"].upper(),
                        "callback_data": distribution["format"].lower(),
                    }
                ]
            )

        response = create_menu(
            conversation_id, "¿En qué formato quieres los datos?", buttons
        )

        if response.ok:
            # print("File format menu rendered.")
            # dispatcher.utter_message(text="Mostrando el menu (file format).")
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            # print("Error rendering menu (file format).")
            dispatcher.utter_message(text="Error al mostrar el menu (file format).")
            return []


class ActionSearchData(Action):
    def name(self) -> Text:
        return "action_search_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        dataset_id = tracker.get_slot("dataset_uuid")
        dataset_meta = tracker.get_slot("data_meta")
        dataset_data = tracker.get_slot("data")
        payload = {}
        headers = {}

        if dataset_id is None:
            dispatcher.utter_message(text="No hay dataset seleccionado.")
            return [SlotSet("menu_message_id", None)]
        elif dataset_meta is not None and dataset_data is not None:
            return [SlotSet("menu_message_id", None)]
            SlotSet("menu_message_id", None)
        else:
            dispatcher.utter_message(
                text="Cargando información sobre el conjunto de datos seleccionado…"
            )
            # Fetch dataset metadata
            url_meta = (
                f"{os.getenv('DKAN_API')}/metastore/schemas/dataset/items/{dataset_id}"
            )

            response_meta = requests.request(
                "GET", url_meta, headers=headers, data=payload
            )

            if response_meta.status_code != 200:
                dispatcher.utter_message(text="Error al cargar los datos del dataset.")
                return [SlotSet("menu_message_id", None)]

            # Find datastore index with csv format
            datastore_index = 0
            dataset_distributions = response_meta.json()["distribution"]
            for idx, item in enumerate(dataset_distributions):
                if item["format"] == "csv":
                    datastore_index = idx

            # Fetch dataset datastore
            url_datastore = f"{os.getenv('DKAN_API')}/datastore/query/{dataset_id}/{datastore_index}?count=true&results=true&schema=true&keys=true&format=json"
            response_datastore = requests.request(
                "GET", url_datastore, headers=headers, data=payload
            )

            if response_datastore.status_code != 200:
                dispatcher.utter_message(
                    text="Error al cargar los datos del datastore."
                )
                return [SlotSet("menu_message_id", None)]

            return [
                SlotSet("menu_message_id", None),
                SlotSet("data_title", response_meta.json()["title"]),
                SlotSet("data_meta", response_meta.json()),
                SlotSet("data", response_datastore.json()),
            ]


class ActionPlotData(Action):
    def name(self) -> Text:
        return "action_plot_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        dispatcher.utter_message(text="Generando gráfico...")

        data = tracker.get_slot("data")
        data_meta = tracker.get_slot("data_meta")
        # print(data)
        # print(data_meta)

        # Create image
        df = pd.json_normalize(data, record_path=["results"])
        # Transform suitable data to numeric or keep as string
        df = df.apply(pd.to_numeric, errors="coerce").fillna(df)

        # df_csv = pd.read_csv(
        #     "http://datos.cadiz.local.ddev.site/sites/default/files/uploaded_resources/residuos-recogidos-por-servicios-municipales-v1.0.0.csv",
        #     delimiter=",",
        # )
        # print("CSV loaded")
        # print("DATAFRAME:", df)
        # print("DATAFRAME:", df.shape)
        # print("DATAFRAME:", df.info())
        # print("CSV:", df_csv)
        # print("CSV:", df_csv.shape)
        # print("CSV:", df_csv.info())

        plt.style.use("ggplot")
        plt.figure(figsize=(10, 5))
        # plt.title(data["query"]["resources"][0]["id"])
        plt.title(data_meta["title"])
        plt.xlabel(data["query"]["properties"][0])

        for property in data["query"]["properties"]:
            # skip first property
            if property == data["query"]["properties"][0]:
                continue
            # remove unwanted properties
            if property in aggregate_titles:
                continue

            # print(property)
            # df[property] = pd.to_numeric(df[property], errors="coerce")
            if pd.api.types.is_numeric_dtype(df[property]):
                plt.plot(
                    df[data["query"]["properties"][0]],
                    df[property],
                    marker=".",
                    markersize=10,
                    label=data["schema"][data["query"]["resources"][0]["id"]]["fields"][
                        property
                    ]["description"],
                )
        plt.ticklabel_format(style="plain", axis="both", useMathText=False)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            frameon=False,
            ncol=3,
            fontsize=8,
        )
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
        # plt.tight_layout()

        img_data = io.BytesIO()
        plt.savefig(img_data, format="png", dpi=72, bbox_inches="tight")
        img_data.seek(0)

        # Create S3 client
        # boto3 currently has a bug with regions launched after 2019
        # this is fixed by setting the endpoint_url in boto3.client
        # https://github.com/boto/boto3/issues/2864
        try:
            # We use boto3.client instead of boto3.resource because the bug
            # is not fixed in boto3.resource
            s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION"),
                endpoint_url=f"https://s3.{os.getenv('AWS_REGION')}.amazonaws.com",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        except ClientError as ce:
            print("error", ce)
        finally:
            try:
                # Save image to S3
                s3_client.put_object(
                    Body=img_data,
                    ContentType="image/png",
                    Bucket=os.getenv("S3_BUCKET_NAME"),
                    Key=f"{conversation_id}.png",
                    Expires=datetime.datetime.now() + datetime.timedelta(days=1),
                )

                # Get a presigned url to avoid public access
                presigned_image_url = s3_client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": f"{os.getenv('S3_BUCKET_NAME')}",
                        "Key": f"{conversation_id}.png",
                    },
                    ExpiresIn=3600,
                )
                dispatcher.utter_message(
                    image=presigned_image_url,
                )
            except ClientError as ce:
                print("error", ce)
            finally:
                s3_client.close()

        return [SlotSet("menu_message_id", None)]


class ActionDownloadData(Action):
    def name(self) -> Text:
        return "action_download_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # theres a bug with rasa use of aiogram in utter_send_file for some file formats,
        # pdf works ok but csv and xlsx don't, so we are using the telegram api directly
        # aiogram.utils.exceptions.WrongFileIdentifier: Wrong file identifier/http url specified
        # once the bug is fixed we can use the dispatcher.utter_message
        # dispatcher.utter_message(response="utter_send_file", file_url=document_url)

        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        data_meta = tracker.get_slot("data_meta")
        data_file_format = tracker.get_slot("data_file_format")
        data_distributions = data_meta["distribution"]

        document_index = None
        for i, obj in enumerate(data_distributions):
            print(i, obj["format"])
            if obj["format"] == data_file_format.lower():
                document_index = i
                break

        print("document_index:", document_index)

        request_url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_API_TOKEN')}/sendDocument"
        document_url = data_meta["distribution"][document_index]["downloadURL"]
        filename = os.path.basename(document_url)
        response_document = requests.get(document_url, verify=True)

        files = {
            "document": (filename, response_document.content),
        }
        params = {"chat_id": conversation_id, "filename": filename}
        response = requests.post(request_url, files=files, params=params)

        if response.ok:
            print(f"{data_file_format} dile sent successfully.")
        else:
            print("Failed to send the CSV file.")
            dispatcher.utter_message(text="Error enviando el archivo.")

        return [SlotSet("menu_message_id", None), SlotSet("data_file_format", None)]


class ActionExplainData(Action):
    def name(self) -> Text:
        return "action_explain_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        data = tracker.get_slot("data")
        print(data["results"])
        prompt = f"""
        Summarize the dataset in json delimited by triple backticks into a single sentence in spanish.
        ```{data["results"]}```
        """
        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_explain_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )

        return [SlotSet("menu_message_id", None)]


class ActionStatisticsData(Action):
    def name(self) -> Text:
        return "action_statistics_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        data = tracker.get_slot("data")

        prompt = f"""
        Summarize the data in json delimited by triple backticks in spanish. Include statistics such as the number of rows and columns, the mean, median, mode, and standard deviation of each column, and the correlation between columns.
        ```{data["results"]}```
        """

        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_statistics_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )
        # dispatcher.utter_message(text="ChatGPT en modo debug (code: 002).")

        return [SlotSet("menu_message_id", None)]


class ActionCustomQueryData(Action):
    def name(self) -> Text:
        return "action_custom_query_data"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        data = tracker.get_slot("data")
        query = tracker.get_slot("data_custom_query")

        prompt = f"""
        I need you to answer a question in spanish about the data in json delimited by triple backticks:
        The question is: {query}
        
        ```{data["results"]}```
        """

        try:
            response = get_completion(
                conversation_id=conversation_id,
                rasa_action="action_custom_query_data",
                prompt=prompt,
            )
            print(response)
            dispatcher.utter_message(text=response)
        except Exception as e:
            dispatcher.utter_message(
                text="Ha ocurrido un error al evaluar los datos, el conjunto de datos es demasiado grande."
            )

        return [SlotSet("data_custom_query", None), SlotSet("menu_message_id", None)]


class ActionRestartCustom(Action):
    def name(self) -> Text:
        return "action_restart_custom"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        dispatcher.utter_message(text="Empecemos de nuevo.")

        return [AllSlotsReset()]


class ActionShowMenu(Action):
    def name(self) -> Text:
        return "action_show_menu"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        # if message_id is not None:
        #     clear_menu(conversation_id, message_id, dispatcher, self.name())

        buttons = [
            [
                {
                    "text": "📈 Mostar gráfico",
                    "callback_data": "/plot_data",
                }
            ],
            [
                {
                    "text": "📄 Descargar archivo",
                    "callback_data": "/download_data",
                }
            ],
            [
                {
                    "text": "ℹ️ Explicar datos",
                    "callback_data": "/explain_data",
                }
            ],
            [
                {
                    "text": "🧮 Mostrar datos estadísticos",
                    "callback_data": "/statistics_data",
                }
            ],
            [
                {
                    "text": "💬 Consulta personalizada",
                    "callback_data": "/custom_query_data",
                }
            ],
            [
                {
                    "text": "🔄 Reiniciar",
                    "callback_data": "/start_over",
                }
            ],
        ]

        response = create_menu(conversation_id, "¿Qué quieres hacer?", buttons)

        if response.ok:
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            # print("Error rendering menu.")
            dispatcher.utter_message(text="Error al mostrar el menu (main).")
            return []


class ActionShowRestartMenu(Action):
    def name(self) -> Text:
        return "action_show_restart_menu"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # Get conversation id
        conversation_id = tracker.sender_id
        message_id = tracker.get_slot("menu_message_id")

        if message_id is not None:
            clear_menu(conversation_id, message_id, dispatcher, self.name())

        buttons = [
            [
                {
                    "text": "Si",
                    "callback_data": "si",
                }
            ],
            [
                {
                    "text": "No",
                    "callback_data": "no",
                }
            ],
        ]

        response = create_menu(conversation_id, "¿Quieres empezar de nuevo?", buttons)

        if response.ok:
            return [SlotSet("menu_message_id", response.json()["result"]["message_id"])]
        else:
            dispatcher.utter_message(text="Error al mostrar el menu. (restart)")
            return []
