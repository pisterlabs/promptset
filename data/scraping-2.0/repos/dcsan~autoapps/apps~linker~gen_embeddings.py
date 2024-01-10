import openai
import os
import json

openai.api_key = os.getenv("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")


expands_prompt = f"""
Given the ideas presented in "{{text}}", write a detailed explanation that expands on these ideas. 
Discuss related concepts, provide more context, and delve deeper into the implications of these ideas.
"""

refines_prompt = f"""
Consider the arguments and points made in "{{text}}". Write a text that refines these arguments, making them more precise and clear. 
Address any ambiguities or generalizations in the original arguments and provide a more nuanced perspective.
"""

contradicts_prompt = f"""
After reviewing the main points of "{{text}}", write a paragraph that presents a counterargument. 
Provide evidence or reasoning that contradicts the arguments or facts presented in the original document.
"""

contextualizes_prompt = f"""
Based on the content of "{{text}}", write a text that provides broader context for these ideas. 
Discuss the historical, cultural, or theoretical background that informs these ideas and how they fit into larger trends or debates.
"""

supports_prompt = f"""
Reflecting on the arguments in "{{text}}", write a text that supports these arguments. 
Provide additional evidence, reasoning, or examples that reinforce the points made in the original document.
"""

parallels = f"""
Taking into account the situation or argument presented in "{{text}}", write a text that presents a similar situation or argument in a different context. 
The new context should be different but the underlying situation or argument should be parallel to the one in the original document.
"""

relation_generation_prompts = {
    "expands": expands_prompt,
    "refines": refines_prompt,
    "contradicts": contradicts_prompt,
    "contextualizes": contextualizes_prompt,
    "supports": supports_prompt,
    "parallels": parallels,
}


def generate_relation_text(text):
    outputs = {}
    for rtype, prompt in relation_generation_prompts.items():
        input_text = prompt.format(text=text)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": input_text}]
        )
        outputs[rtype] = response["choices"][0]["message"]["content"]
    return outputs


if __name__ == "__main__":
    # text = """
    # It all starts with the universally applicable premise that people want to be understood and accepted. Listening is the cheapest, yet most effective concession we can make to get there. By listening intensely, a negotiator demonstrates empathy and shows a sincere desire to better understand what the other side is experiencing.
    # """.strip()
    with open("./data/readwise_database_KnowledgeAgent.json") as g:
        all_highlights = json.load(g)

    all_generations_per_hl = []
    for h in all_highlights[0]["highlights"]:
        text = h["text"]
        outputs = generate_relation_text(text) # relation type -> generated_text
        all_generations_per_hl.append(outputs)

    with open("all_generations.json", "w") as f:
        json.dump(all_generations_per_hl, f)
    # transform output to nodes viz format

    # print(outputs)


"""
[
  {
    "nodes": [
      {
        "id": "node1",
        "text": "It all starts with the universally applicable premise that people want to be understood and accepted. Listening is the cheapest, yet most effective concession we can make to get there. By listening intensely, a negotiator demonstrates empathy and shows a sincere desire to better understand what the other side is experiencing.",
        "keywords": [
          "red",
          "green"
        ]
      },
      {
        "id": "node2",
        "text": "Effective negotiation is applied people smarts, a psychological edge in every domain of life: how to size someone up, how to influence their sizing up of you, and how to use that knowledge to get what you want.",
      }
    ]
  },
  {
    "edges": [
      {
        "from": "node1",
        "to": "node2",
        "type": "similarity",
        "description": "some more on the relation between 1 and 2"
      },
    ]
  }
]
"""
