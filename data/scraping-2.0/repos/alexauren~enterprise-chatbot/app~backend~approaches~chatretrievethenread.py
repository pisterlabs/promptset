import time
import re
import concurrent.futures
from typing import Any, Sequence
import openai
import openai.error
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

class ChatRetrieveThenReadApproach(Approach):
    """
    Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
    top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
    (answer) with that prompt.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    DOCUMENT_SCORE_CUTOFF = 1

    CHATGPT_TIMEOUT = 600
    CHATGPT_RETRY_WAIT = 1
    CHATGPT_MAX_RETRIES = 3


    assistant_prompt = """
Your name is Floyd and you are a helpful insurance customer assistant representing DNB bank ASA. You respond with the same language as the question wes asked. Be brief in your answers. If the user asks something unrelated to DNB insurance, say that you can't answer that.
Answer ONLY with the facts listed in the list of sources below ```Sources```. If there isn't enough information below or the answer is not related to the sources, say you don't know. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
When asked a question and there are no sources available, tell the customer that you unfortunately cant answer that, as its not in your sources but that the customer may find information here: https://www.dnb.no/en/insurance. 

When asked a question you have been asked earlier in the chat, tell the customer the same thing as earlier, or tell them to be more specific please
Examples:
User: Does DNB offer house insurance?
Assistent: DNB does offer house insurance [Source 1]
User: What is the price of the house insurance?
Assistent: That depends on several factors, allow us to calculate how much insurance is going to cost you by going to our website. [Source 1]
User: What is the difference between a cat and a dog?
Assistent: Unfortunately I cant answer that, as its not in the sources I have been given, please ask something related to house or content insurance. Check out https://www.dnb.no/en/insurance for more information
User: What is Kasko?
Assistent: Unfortunately I cant answer that, as its not in the sources I have been given, please ask something related to house or content insurance. Check out https://www.dnb.no/en/insurance for more information
{follow_up_questions_prompt}
{injected_prompt}
```Sources```
{sources}
"""
    no_source = """ 
    You are helpful insurance customer assistant representing DNB bank ASA. Please refere to DNB as your insurance company. You respond with the same language as the question asked.
    You must say that you cant answer this question in the same language as the question: {question}.
    Please answer the client in the same way as below: 
    Assistant: Sorry, I cant answer the question becuase I cant find any relevant sources.
    Assistant: Beklager, jeg kan ikke svare på spørsmålet fordi jeg skrånende finner noen relevante kilder.
    Assistant: Извините, я не могу ответить на вопрос, потому что не могу найти соответствующие источники.
    """
    

    query_prompt = """Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about DNB insurance.
Generate a search query based on the conversation and the new question. 
Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
Do not include any special characters like '+'.
It is important that the search query is in english such that cognitive search can search efficient you can use the example below:
"Query: Hva er husforsikring?
Search query: What is house insurance?
Query: ¿Cuánto cuesta el seguro de automóvil? "
Search query: What does car insurance cost?
Query: Was ist der Unterschied zwischen Hausratversicherung und Hausratversicherung?
Search query: What is the difference between contents insurance and home insurance


History:
{history}
"""


    query_prompt_few_shots = [
        {'role' : USER, 'content' : 'What house insurance does DNB provide?' },
        {'role' : ASSISTANT, 'content' : 'house insurance types' },
        {'role' : USER, 'content' : 'What does standard house insurance cover?' },
        {'role' : ASSISTANT, 'content' : 'standard house insurance coverage' }
    ]

    follow_up_questions_prompt_content = """After giving your answer, generate three very brief follow-up questions that the user would likely ask next.
    Base your questions on the sources used in the previous answer if there are any sources there.
    Try not to repeat questions that have already been asked.
    Only generate questions and do not generate any text before or after the questions, such as 'Next Questions'
    Use double angle brackets to reference the questions.
    Format:
    <<What is the cheapest alternative?>> <<What does it cover?>> <<How much does it cost?>>"""


    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.chatgpt_deployment = chatgpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.executor = concurrent.futures.ThreadPoolExecutor()
    
    def run(self, history: Sequence[dict[str, str]], overrides: dict[str, Any]) -> Any:
        start_time = time.time()

        print("Starting answering process")
        print(f"Max time limit for chatGPT has been set to {self.CHATGPT_TIMEOUT} seconds")

        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 6
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
    
        print("Beginning step 1: Generate keyword search query")

        filtered_history = self.clear_history(history)
        
        step_time = time.time()
        search_query = self.generate_keyword_query(filtered_history, overrides, self.CHATGPT_TIMEOUT)
        print(f"Finished step 1 in {time.time() - step_time} seconds")

        if search_query == None:
            return {"data_points": "", "answer": "Could not generate query, please try again.", "thoughts": ""}

        print(f" Original search query: {search_query}")
      
        print("Beginning step 2: Retrieve documents from search index")

        step_time = time.time()
        documents = self.retrieve_documents(search_query, top, filter, use_semantic_captions, overrides)
        source_list = self.documents_to_sources(documents, use_semantic_captions)
        sources = len(source_list) and "\n".join(source_list) or ""

        print(f"Finished step 2 in {time.time() - step_time} seconds")
        print("Beginning step 3: Generate question answer")

        step_time = time.time()
        prompt = self.format_assistant_prompt(sources, overrides)
        answer = self.generate_question_answer(prompt, filtered_history, overrides, self.CHATGPT_TIMEOUT)
        if answer == None:
            print("WARNING: Timeout before generating question answer")
            answer = "Sorry, I can't answer the question."
            # answer = self.generate_question_answer(self.no_source, filtered_history[len(filtered_history)], overrides, self.CHATGPT_TIMEOUT)
         
            

        print("Generated answer: ", answer)

        if not self.check_answer_sources(answer, documents, filtered_history):
            print("WARNING: Generated question answer used sources incorrectly")
            answer = "Sorry, I do not have information related to your question."
            # prompt = self.no_source.format(question=filtered_history[-1])
            # answer = self.generate_question_answer(prompt,[], overrides, self.CHATGPT_TIMEOUT)
        


        print(f"Finished step 3 in {time.time() - step_time} seconds")
        print(f"Answering process completed in {time.time() - start_time} seconds")

        thoughts = f"Searched for:<br>{search_query}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')
        if overrides.get("suggest_followup_questions"):
            answer = self.remove_wrong_questions_format(answer,"Next Questions: ")

        
        return {"data_points": source_list, "answer": answer, "thoughts": thoughts}

    def generate_keyword_query(self, history, overrides, timeout):
        user_question = f"Generate search query for: {history[-1][self.USER]}"
        prompt = self.query_prompt.format(history=self.history_as_text(history[:-1]))
        messages = self.format_chat_messages(system_prompt=prompt, history=[], user_question=user_question, few_shot=self.query_prompt_few_shots)
        future = self.executor.submit(self.get_completion, messages, overrides)
        try:
            completion = future.result(timeout=timeout)
            return completion.choices[0].message.content
        except concurrent.futures.TimeoutError:
            return None

    def retrieve_documents(self, query, top, filter, use_semantic_captions, overrides):
        if overrides.get("semantic_ranker"):
            r = self.search_client.search(query, 
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="en-us", 
                                          query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top=top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        
        else:
            r = self.search_client.search(query, filter=filter, top=top)

        documents = []
        for doc in r:
            score = doc["@search.score"]
            if score < self.DOCUMENT_SCORE_CUTOFF:
                print(f"Removed doc {doc[self.sourcepage_field]} with score {score}")
            else:
                print(f"Kept doc {doc[self.sourcepage_field]} with score {score}")
                documents.append(doc)
        
        return documents

    def documents_to_sources(self, documents, use_semantic_captions):
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in documents]
        else:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(doc[self.content_field]) for doc in documents]

        return results

    def check_answer_sources(self, answer, documents, history):
        source_regex = r"\[([^]]+)\]"
        answer_sources = re.findall(source_regex, answer)
        search_documents = [doc[self.sourcepage_field] for doc in documents]
        history_documents = [src for msg in history if self.ASSISTANT in msg for src in re.findall(source_regex, msg[self.ASSISTANT])]

        print("Answer sources: ", answer_sources)
        print("Documents from search: ", search_documents)
        print("Documents from history: ", history_documents)

        for source in answer_sources:
            if source not in search_documents and source not in history_documents:
                print(f"Tried to use incorrect source {source} in answer")
                return False

        return True

    def format_assistant_prompt(self, sources, overrides):
        follow_up_questions_prompt = self.follow_up_questions_prompt_content if overrides.get("suggest_followup_questions") else ""
        
        # Allow client to replace the entire prompt, or to inject into the exiting prompt using >>>
        prompt_override = overrides.get("prompt_template")
        if prompt_override is None:
            prompt = self.assistant_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt, injected_prompt="", sources=sources)
        elif prompt_override.startswith(">>>"):
            prompt = self.assistant_prompt.format(follow_up_questions_prompt=follow_up_questions_prompt, injected_prompt=prompt_override[3:] + "\n", sources=sources)
        else:
            prompt = prompt_override.format(follow_up_questions_prompt=follow_up_questions_prompt, sources=sources)

        return prompt

    def generate_question_answer(self, prompt, history, overrides, timeout):
        messages = self.format_chat_messages(system_prompt=prompt, history=history, user_question=history[-1][self.USER])
        future = self.executor.submit(self.get_completion, messages, overrides)
        try:
            completion = future.result(timeout=timeout)
            if completion:
                return completion.choices[0].message.content
            return None
        except concurrent.futures.TimeoutError:
            return None
    
    def get_completion(self, messages, overrides):
        retries = 0
        while retries <= self.CHATGPT_MAX_RETRIES:
            if retries > 0:
                print(f"Completion failed. Retry number {retries}")

                # Wait a bit before retrying
                time.sleep(self.CHATGPT_RETRY_WAIT)

            try:
                completion = openai.ChatCompletion.create(
                engine=self.chatgpt_deployment,
                messages=messages,
                temperature=overrides.get("temperature") or 0,
                max_tokens=1024,
                n=1,
                )

                return completion
            except openai.error.Timeout as e:
                print(f"OpenAI API request timed out: {e}")
            except openai.error.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")

            retries += 1

        return None


    def format_chat_messages(self, system_prompt: str, history: Sequence[dict[str, str]], user_question: str, few_shot: Sequence[dict[str, str]] = []):
        messages = [{"role": self.SYSTEM, "content": system_prompt}]

        for shot in few_shot:
            messages.append({"role": self.SYSTEM, "name": f"example_{shot.get('role')}", "content": shot.get("content")})

        if len(history) > 0:
            for interaction in history[:-1]:
                for role, content in interaction.items():
                    messages.append({"role": role, "content": content})

        messages.append({"role": self.USER, "content": user_question})

        return messages

    def history_as_text(self, history):
        text = "" 
        for interaction in history:
            for role, content in interaction.items():
                text = "\n".join([text, f"{role}: {content}"])

        return text

    def clear_history(self, history):
        filtered_history = []

        for entry in history:
            if 'assistant' not in entry or ']' in entry['assistant']:
                filtered_history.append(entry)

        return filtered_history
    
    def remove_wrong_questions_format(self, answer, substring):
        print("Checking for wrong format in suggested answers...")
        new_answer = answer.replace(substring, "")
        if  (new_answer != answer):
            print(f"Removed {substring} from answer")
        return new_answer