from chaincrafter import Chain, Prompt, response_format_list, response_style, response_length, extract_items_from_list
from chaincrafter.models import OpenAiChat

gpt4_chat = OpenAiChat(temperature=0.7, model_name="gpt-4")
system_prompt = Prompt("You are a helpful assistant")
example_prompt = Prompt(
    "Tell me three interesting facts about {topic}",
    [
        # Commonly used instructions or prompts can be added as modifiers and re-used
        response_style("professional economist"),
        response_length("long", "answer"),
        response_format_list("Interesting fact"),
    ],
    # The input variables that the prompt expects, and their types (or processing function)
    topic=str
)
followup_question_prompt = Prompt(
    "Could you tell me more about {facts_list}?",
    [
        response_style("a pirate who has just returned from the Galapagos Islands"),
        response_length("short", "2-3 paragraphs"),
    ],
    # Parses and extracts data from the previous response to populate the input variable that is used by the prompt
    facts_list=lambda facts_list: extract_items_from_list(facts_list)[0],
)
chain = Chain(
    system_prompt,
    # The prompt message to send and the output key to store the response in
    (example_prompt, "facts_list"),
    (followup_question_prompt, "output"),
)
messages = chain.run(gpt4_chat, {"topic": "Trees and their positive ecological impact" })
for message in messages:
    print(f"{message['role']}: {message['content']}")
    print()
