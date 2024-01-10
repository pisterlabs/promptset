import redis
import uuid
import json
import openai
from decouple import AutoConfig


config = AutoConfig()
redis_client = redis.Redis(
    host=config("REDIS_HOST"), port=config("REDIS_PORT"), db=config("REDIS_DB")
)
openai.api_key = config("OPENAI_API_KEY")


def generate_unique_id():
    return str(uuid.uuid4())


def add_todo_item(session_id, todo_item_data):
    # Store todo item using the session_id as the key
    redis_client.hset(
        f"todo_items: {session_id}", generate_unique_id(), json.dumps(todo_item_data)
    )


def get_todo_items(session_id):
    # Retrieve all todo items for a specific id
    todo_items = redis_client.hgetall(f"todo_items: {session_id}")

    # Convert the todo item data from JSON strings to dictionaries
    todo_items = {
        k.decode("utf-8"): json.loads(v.decode("utf-8")) for k, v in todo_items.items()
    }

    return todo_items


def update_todo_item_ai(session_id, todo_item_id, todo_item_data):
    # Update a todo item for a specific id with AI help
    todo_item_data["ai_help"] = todo_item_help(todo_item_data["item_name"])
    redis_client.hset(
        f"todo_items: {session_id}", todo_item_id, json.dumps(todo_item_data)
    )


def update_todo_item_complete(session_id, todo_item_id, todo_item_data):
    # Update a todo item for a specific id as completed
    redis_client.hset(
        f"todo_items: {session_id}", todo_item_id, json.dumps(todo_item_data)
    )


def todo_item_help(prompt):
    # Generate help for a todo item
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. You read the user's todo item and help them with their tasks by suggesting actions they can take.",
            },
            {
                "role": "user",
                "content": "Buy milk",
            },
            {
                "role": "assistant",
                "content": "Find the nearest grocery store and buy milk.",
            },
            {
                "role": "user",
                "content": "Repair my phone screen",
            },
            {
                "role": "assistant",
                "content": "Find the nearest phone repair shop and repair my phone screen.",
            },
            {
                "role": "user",
                "content": "Pay rent",
            },
            {
                "role": "assistant",
                "content": "Pay rent on time to avoid late fees.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]

    return result
