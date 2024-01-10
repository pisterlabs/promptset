import os
import openai

from louis.prompts import PROMPTS

def nonewlines(s: str) -> str:
    return s.replace('\n', ' ').replace('\r', ' ')

AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_SERVICE") or "myopenai"
AZURE_OPENAI_GPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_GPT_DEPLOYMENT") or "davinci"
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT") or "chat"

class Louis:
    prompt_prefix = """<|im_start|>system
{louis}
{answer}
{clarification}
{format}
{follow_up_questions_prompt}
{injected_prompt}
Sources:
{sources}
<|im_end|>
{chat_history}
"""

    query_prompt_template = PROMPTS['query_prompt_template']

    def __init__(self, search_client, chatgpt_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT, gpt_deployment=AZURE_OPENAI_GPT_DEPLOYMENT):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question
        prompt = self.query_prompt_template.format(
            chat_history=self.get_chat_history_as_text(history, include_last_turn=False),
            question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment,
            prompt=prompt,
            temperature=0.0,
            max_tokens=32,
            n=1,
            stop=["\n"])
        q = completion.choices[0].text

        sources = self.search_client(q)

        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.prompt_prefix.format(
                injected_prompt="",
                sources=sources,
                chat_history=self.get_chat_history_as_text(history),
                **PROMPTS)
        elif prompt_override.startswith(">>>"):
            prompt = self.prompt_prefix.format(
                injected_prompt=prompt_override[3:] + "\n",
                sources=sources,
                chat_history=self.get_chat_history_as_text(history),
                **PROMPTS)
        else:
            prompt = prompt_override.format(
                sources=sources,
                chat_history=self.get_chat_history_as_text(history),
                **PROMPTS)

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        completion = openai.Completion.create(
            engine=self.chatgpt_deployment,
            prompt=prompt,
            temperature=overrides.get("temperature") or 0.7,
            max_tokens=1024,
            n=1,
            stop=["<|im_end|>", "<|im_start|>"])

        retvalue = {
            "data_points": sources,
            "answer": completion.choices[0].text,
            "thoughts": f"Searched for:<br>{q}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')
        }
        return retvalue

    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break
        return history_text