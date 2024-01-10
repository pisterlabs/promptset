import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import aiosqlite
import dotenv
import openai
import translators as ts
from datasets import Dataset, load_dataset
from prompts import TRANSLATE_JSON

FROM_LANGUAGE, TO_LANGUAGE = "en", "ru"
MAX_CONCURRENCY = 3
TS_ENGINE = "google"
DB_FILE = "data.db"
DIALOG_SUMMARIZATION_DATASETS = [
    "DialogSum",
    "AMI",
    "CRD3",
    "ECTSum",
    "ICSI",
    "MediaSum",
    "QMSum",
    "SAMSum",
    "TweetSumm",
    "ConvoSumm",
    "SummScreen_ForeverDreaming",
    "SummScreen_TVMegaSite",
]
API_KEYS = []
BLACKLISTED_KEYS = set()
CURRENT_PATH = Path(__file__).resolve().parent


# Initialize OpenAI API
def initialize_openai_api():
    api_keys_file = "api_keys.txt"

    keys_path = CURRENT_PATH / api_keys_file
    if not keys_path.is_file():
        print(f"API keys file '{keys_path}' not found.")
        sys.exit(1)

    with open(keys_path, "r") as keys_file:
        api_keys = keys_file.read().splitlines()

    return api_keys


def translate(
    text: str,
    from_lang: str = FROM_LANGUAGE,
    to_lang: str = TO_LANGUAGE,
    ts_engine: str = TS_ENGINE,
) -> str:
    """Translates text from one language to another using translators library.

    Args:
        text (str): Text to translate.
        from_lang (str): Source language code.
        to_lang (str): Target language code.
        ts_engine (str): translators library translation engine to use.

    Returns:
        str: Translated text.
    """
    return ts.translate_text(text, ts_engine, from_lang, to_lang) if text else ""


def get_random_api_key(prev_key):
    # Exclude blacklisted keys
    available_keys = [
        key for key in API_KEYS if key not in BLACKLISTED_KEYS.union(set([prev_key]))
    ]
    if not available_keys:
        print("All API keys are blacklisted.")
        sys.exit(1)

    return random.choice(available_keys)


async def translate_chatgpt(
    text: str, retry_attempt: int = 0, prev_key: str = ""
) -> str:
    """Translates text from English to Russian using ChatGPT API.

    Args:
        text (str): Text to translate.

    Returns:
        str: Translated text.
    """
    try:
        if retry_attempt != 0:
            print(f"OpenAI retry attempt: {retry_attempt}")

        token = get_random_api_key(prev_key)
        openai.api_key = token
        messages = [
            {
                "content": "Translate this text from English to Russian. Write only in Russian. Text:",
                "role": "system",
            },
            {"role": "user", "content": text},
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        return response.choices[0].message.get("content", "")
    except openai.error.RateLimitError as e:
        if retry_attempt > 1:
            await asyncio.sleep(20)
        return await translate_chatgpt(text, retry_attempt + 1, token)
    except openai.error.AuthenticationError as e:
        print(f"API key is incorrect or blacklisted: {token}")
        BLACKLISTED_KEYS.add(token)
        with open(CURRENT_PATH / "blacklisted_keys.txt", "w") as file:
            for key in BLACKLISTED_KEYS:
                file.write(key + "\n")
        return await translate_chatgpt(text, retry_attempt + 1, token)


async def translate_json_chatgpt(
    json_str: str, retry_attempt: int = 0, prev_key: str = ""
) -> str:
    """Translates JSON from English to Russian using ChatGPT API.

    Args:
        text (str): stringified JSON to translate.

    Returns:
        str: Translated stringified JSON.
    """
    try:
        token = get_random_api_key(prev_key)

        if retry_attempt != 0:
            print(f"(JSON - {token}) OpenAI retry attempt: {retry_attempt}")

        openai.api_key = token
        messages = [
            {
                "content": TRANSLATE_JSON.format(json_str),
                "role": "system",
            },
        ]
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = response.choices[0].message.get("content", "")
        try:
            _ = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"Input string is not valid JSON. Error: {e}")
            return await translate_json_chatgpt(json_str, retry_attempt + 1, token)
        return result
    except openai.error.RateLimitError as e:
        if retry_attempt > 1:
            await asyncio.sleep(2)
        return await translate_json_chatgpt(json_str, retry_attempt + 1, token)
    except openai.error.AuthenticationError as e:
        print(f"API key is incorrect or blacklisted: {token}")
        BLACKLISTED_KEYS.add(token)
        with open(CURRENT_PATH / "blacklisted_keys.txt", "w") as file:
            for key in BLACKLISTED_KEYS:
                file.write(key + "\n")
        return await translate_json_chatgpt(json_str, retry_attempt + 1, token)


async def create_db_table():
    """Creates SQLite database to store translation data."""
    conn = await aiosqlite.connect(DB_FILE)
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dialogsum_translation
            (original_dialog_id TEXT, 
            new_dialog_id TEXT, 
            dialog_index INT, 
            original_dialog_info TEXT,  
            log TEXT, 
            prompt TEXT,
            translated_original_dialog_info TEXT,
            translated_log TEXT,
            chatgpt_translated_original_dialog_info TEXT,
            chatgpt_translated_log TEXT,
            split TEXT,
            status TEXT)
        """
    )
    await conn.commit()
    await conn.close()


async def upload_dataset(dataset: Dataset):
    """Uploads dataset into database."""
    await create_db_table()
    conn = await aiosqlite.connect(DB_FILE)
    added_rows_count = 0
    for split in dataset:
        added_rows_count += len(dataset[split])
        for row in dataset[split]:
            # Dump all data to JSON
            dataset_values = [
                json.dumps(item) if not isinstance(item, str) else item
                for item in list(row.values())
            ]
            values = dataset_values + ["", "", "", "", str(split), "pending"]
            placeholders = ", ".join(["?" for _ in values])
            await conn.execute(
                f"INSERT INTO dialogsum_translation VALUES ({placeholders})",
                values,
            )
    await conn.commit()
    await conn.close()

    print(f"Added rows count: {added_rows_count}")


async def get_row_by_new_dialog_id(new_dialog_id: str) -> [dict, None]:
    """Retrieves a single row from database by new_dialog_id.

    Args:
        new_dialog_id (str): new_dialog_id value to query

    Returns:
        dict: Row data
    """
    try:
        async with aiosqlite.connect(DB_FILE) as conn:
            async with conn.execute(
                "SELECT * FROM dialogsum_translation WHERE new_dialog_id = ?",
                [
                    new_dialog_id,
                ],
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    column_labels = [
                        description[0] for description in cursor.description
                    ]
                    result_dict = dict(zip(column_labels, row))
                    return result_dict
                else:
                    print("Row not found")
    except Exception as e:
        print("ERROR: ", e)
        raise e


def translate_original_dialog_info(row_dict: dict) -> dict:
    """Function for translating original_dialog_info column

    Args:
        row_dict (dict): Original values from DB rows

    Returns:
        dict: DB row filled with translations
    """
    if row_dict["translated_original_dialog_info"] != "":
        return row_dict
    # Parse JSON
    original_dialog_info = json.loads(row_dict["original_dialog_info"])
    # Translate using API
    translated_original_dialog_info = dict()
    translated_original_dialog_info["summary"] = translate(
        original_dialog_info["summary"]
    )
    translated_original_dialog_info["topic"] = translate(original_dialog_info["topic"])
    # Dumps JSON
    row_dict["translated_original_dialog_info"] = json.dumps(
        translated_original_dialog_info, ensure_ascii=False
    )
    return row_dict


async def chatgpt_translate_original_dialog_info(row_dict: dict) -> dict:
    """Function for translating original_dialog_info column

    Args:
        row_dict (dict): Original values from DB rows

    Returns:
        dict: DB row filled with translations
    """
    if row_dict["chatgpt_translated_original_dialog_info"] != "":
        return row_dict
    # Parse JSON
    original_dialog_info = json.loads(row_dict["original_dialog_info"])
    # Translate using OpenAI API
    chatgpt_translated_original_dialog_info = dict()
    chatgpt_translated_original_dialog_info["summary"] = await translate_chatgpt(
        original_dialog_info["summary"]
    )
    chatgpt_translated_original_dialog_info["topic"] = await translate_chatgpt(
        original_dialog_info["topic"]
    )
    # Dumps JSON
    row_dict["chatgpt_translated_original_dialog_info"] = json.dumps(
        chatgpt_translated_original_dialog_info, ensure_ascii=False
    )

    # TODO: work on optimization
    # row_dict["chatgpt_translated_original_dialog_info"] = await translate_json_chatgpt(
    #     row_dict["original_dialog_info"]
    # )
    return row_dict


def translate_log(row_dict: dict) -> dict:
    """Translates log column using translators library.

    Args:
        row_dict (dict): Row data

    Returns:
        dict: Updated row data with translations
    """

    if row_dict["translated_log"] != "":
        return row_dict

    log = json.loads(row_dict["log"])

    translated_log = []
    history = ""

    for i, turn in enumerate(log):
        translated_turn = {}

        for k, v in turn.items():
            if k == "user utterance" or k == "system response":
                translated_v = translate(v)
            else:
                translated_v = v

            translated_turn[k] = translated_v

            if k == "dialog history":
                if i == 0:
                    continue
                history += f"<USER> {translated_log[i-1]['user utterance']} "
                history += f"<SYSTEM> {translated_log[i-1]['system response']} "
                translated_turn[k] = history

        translated_log.append(translated_turn)

    row_dict["translated_log"] = json.dumps(translated_log, ensure_ascii=False)

    return row_dict


async def chatgpt_translate_log(row_dict: dict) -> dict:
    """Translates log column using ChatGPT API.

    Args:
        row_dict (dict): Row data

    Returns:
        dict: Updated row data with translations
    """
    if row_dict["chatgpt_translated_log"] != "":
        return row_dict

    log = json.loads(row_dict["log"])

    chatgpt_translated_log = []
    history = ""

    for i, turn in enumerate(log):
        chatgpt_translated_turn = {}
        for k, v in turn.items():
            if k == "user utterance" or k == "system response":
                chatgpt_translated_v = await translate_chatgpt(v)
            else:
                chatgpt_translated_v = v

            chatgpt_translated_turn[k] = chatgpt_translated_v

            if k == "dialog history":
                if i == 0:
                    continue
                history += f"<USER> {chatgpt_translated_log[i-1]['user utterance']} "
                history += f"<SYSTEM> {chatgpt_translated_log[i-1]['system response']} "
                chatgpt_translated_turn[k] = history

        chatgpt_translated_log.append(chatgpt_translated_turn)

    row_dict["chatgpt_translated_log"] = json.dumps(
        chatgpt_translated_log, ensure_ascii=False
    )

    # TODO: work on optimization
    # row_dict["chatgpt_translated_log"] = await translate_json_chatgpt(row_dict["log"])

    return row_dict


async def update_row(new_dialog_id: str, data: dict):
    """Updates a database row.

    Args:
        new_dialog_id (str): id of row to update
        data (dict): updated data values

    """
    conn = await aiosqlite.connect(DB_FILE)

    set_clause = ", ".join([f"{key} = ?" for key in data.keys()])
    values = [data[key] for key in data.keys()]
    values.append(new_dialog_id)

    query = f"UPDATE dialogsum_translation SET {set_clause} WHERE new_dialog_id = ?"

    await conn.execute(query, values)
    await conn.commit()

    await conn.close()


async def row_translating_task(
    new_dialog_id: str, semaphore: asyncio.Semaphore
) -> bool:
    """Performs full translation workflow for a single row.

    Args:
        new_dialog_id (str): id of row to process
        semaphore (asyncio.Semaphore): semaphore border

    Returns:
        bool: True if successful otherwise False
    """
    async with semaphore:
        try:
            print(f"Start task for {new_dialog_id}")
            start_time = time.time()
            row_dict = await get_row_by_new_dialog_id(new_dialog_id)
            if row_dict["status"] == "translated":
                return True
            print(f"[Step 0/5] Got row by new_dialog_id ({time.time() - start_time}s)")
            row_dict = translate_original_dialog_info(row_dict)
            print(f"[Step 1/5] Done for {new_dialog_id} ({time.time() - start_time}s)")
            row_dict = translate_log(row_dict)
            await update_row(new_dialog_id, row_dict)
            print(f"[Step 2/5] Done for {new_dialog_id} ({time.time() - start_time}s)")
            row_dict = await chatgpt_translate_original_dialog_info(row_dict)
            print(f"[Step 3/5] Done for {new_dialog_id} ({time.time() - start_time}s)")
            row_dict = await chatgpt_translate_log(row_dict)
            print(f"[Step 4/5] Done for {new_dialog_id} ({time.time() - start_time}s)")
            row_dict["status"] = "translated"
            await update_row(new_dialog_id, row_dict)
            print("---")
            print(f"[Step 5/5] Done for {new_dialog_id} ({time.time() - start_time}s)")
            print("---")
            return True
        except Exception as e:
            print(e)
            return False


async def translate_rows():
    conn = await aiosqlite.connect(DB_FILE)

    db_exists = os.path.exists(DB_FILE)

    if db_exists:
        conn = await aiosqlite.connect(DB_FILE)

        # Check if table exists
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dialogsum_translation'"
        )
        table_exists = await cursor.fetchone() is not None

        # Check if table is empty
        if table_exists:
            cursor = await conn.execute("SELECT COUNT(*) FROM dialogsum_translation")
            num_rows = await cursor.fetchone()
            table_empty = num_rows[0] == 0

    if not table_exists or table_empty:
        print("Table does not exist or is empty, creating and uploading dataset...")

        dataset = load_dataset(
            "Salesforce/dialogstudio", DIALOG_SUMMARIZATION_DATASETS[0]
        )
        await upload_dataset(dataset)

    async with conn.execute(
        "SELECT new_dialog_id FROM dialogsum_translation WHERE status = 'pending'"
    ) as cursor:
        rows = await cursor.fetchall()

    await conn.close()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = []
    for row in rows:
        task = asyncio.create_task(row_translating_task(row[0], semaphore))
        tasks.append(task)

    print(f"Tasks in progress: {len(tasks)}")
    await asyncio.gather(*tasks)


def stop_loop(loop):
    loop.stop()


if __name__ == "__main__":
    API_KEYS = initialize_openai_api()

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        mode = "Run"
    else:
        loop = asyncio.get_event_loop()
        mode = "Reader"

    loop.run_until_complete(translate_rows())
    loop.close()
