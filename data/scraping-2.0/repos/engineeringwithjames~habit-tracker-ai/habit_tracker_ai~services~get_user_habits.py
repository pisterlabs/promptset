from langchain.agents import tool
from config import db


def get_user_habits(user_id: str) -> str:
    """Get the user's existing habits from the database."""
    habit_docs = db.collection('habits').where(
        'userId', '==', user_id).stream()
    habit = []

    for doc in habit_docs:
        habit.append(doc.to_dict()["name"])

    if len(habit) == 0:
        return "No habits found."

    return ",".join(habit)
