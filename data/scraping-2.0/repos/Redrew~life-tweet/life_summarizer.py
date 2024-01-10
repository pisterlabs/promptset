from openai_api import *
from evaluate_web_browser_history import evaluate_browser_history

def add_deltas(profile, summary, deltas):
    summary = summary.strip()
    relevancy = evaluate_browser_history(summary, engine)
    for category, relevant in relevancy.items():
        if relevant == 'no': continue
        traits = profile[category]
        biographer_info = f"You are a biographer writing about Alex's hobbies.\nYou know this about Alex. {traits}\nAlex is reading a website. The website says:\n{summary}"
        scores = {}
        for label, statement in [
            ("yes", biographer_info + f" You learned something new about Alex's {category}"),
            ("no", biographer_info + f" You didn't learned something new about Alex's {category}"),
        ]:
            scores[label] = engine.score(statement)
        learned_something_new = max(scores.keys(), key=scores.get)
        if learned_something_new == "no": continue
        new_info = get_chat_gpt_output(biographer_info + " You learned something new about Alex today. You learned that ").strip()
        deltas[category].append(new_info)

    return deltas