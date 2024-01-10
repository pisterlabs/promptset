import openai
# import time
import backoff


class LLMUtil:
    OPENAI_API_KEY = "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"

    DAVINCI_MODEL_NAME = "text-davinci-003"
    TURBO_MODEL_NAME = "gpt-3.5-turbo"
    GPT4_MODEL_NAME = "gpt-4"
    TURBO_0301_MODEL_NAME = "gpt-3.5-turbo-0301"
    ROLE_SYSTEM = 'system'
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    # ANSWER_SEQUENCE = "\nAI: "
    # QUESTION_SEQUENCE = "\n\nKG: "
    # ANSWER_SEQUENCE = "\nA: "
    # QUESTION_SEQUENCE = "\n\nQ: "
    SESSION_PROMPT = ""
    CHAT_LOG = None

    GPT3 = 'GPT-3'
    CHATGPT = 'chatGPT'

    # @staticmethod
    # def ask_davinci(question, chat_log=None):
    #     """
    #     ask LLM question
    #     Args:
    #         question (question):
    #         chat_log (QA history):
    #
    #     Returns: answer
    #
    #     """
    #     if chat_log is None and LLMUtil.SESSION_PROMPT is not None:
    #         chat_log = LLMUtil.SESSION_PROMPT
    #     prompt_text = f'{chat_log}{LLMUtil.QUESTION_SEQUENCE}{question}{LLMUtil.ANSWER_SEQUENCE}'
    #     # print(prompt_text)
    #     response = openai.Completion.create(
    #         model=LLMUtil.DAVINCI_MODEL_NAME,
    #         prompt=prompt_text,
    #         temperature=0.7,
    #         max_tokens=60,
    #         top_p=1.0,
    #         frequency_penalty=0.0,
    #         presence_penalty=0.0,
    #         # stop=[" KG:", " AI:"]
    #     )
    #     answer = response['choices'][0]['text'].strip()
    #     chat_log = LLMUtil.append_interaction_to_chat_log(question, answer, chat_log)
    #     return prompt_text, answer, chat_log

    # @staticmethod
    # def append_interaction_to_chat_log(question, answer, chat_log=None):
    #     """
    #     append the QA pair to chat_log
    #     Args:
    #         question ():
    #         answer ():
    #         chat_log ():
    #
    #     Returns: updated chat_log
    #
    #     """
    #     if chat_log is None:
    #         chat_log = LLMUtil.SESSION_PROMPT
    #     return f'{chat_log}{LLMUtil.QUESTION_SEQUENCE}{question}{LLMUtil.ANSWER_SEQUENCE}{answer}'
    #
    # @staticmethod
    # def question_answer(question):
    #     """
    #     ask AI question and get answer
    #     Args:
    #         question ():
    #
    #     Returns: answer
    #     Note: dict 引用传递
    #     """
    #     # question = question_dict[dict_key]
    #     # print(f"{LLMUtil.QUESTION_SEQUENCE}{question}")
    #     prompt_text, answer, LLMUtil.CHAT_LOG = LLMUtil.ask_davinci(question, LLMUtil.CHAT_LOG)
    #     # answer = answer.strip()
    #     # print(f"{LLMUtil.ANSWER_SEQUENCE}{answer}")
    #     return prompt_text, answer
    #     # answer_dict[dict_key] = answer
    #     # return answer_dict
    #
    # @staticmethod
    # def reach_chatgpt_limited_request(answer, summary_pair):
    #     if answer == "Unusable response produced, maybe login session expired. " \
    #                  "Try 'pkill firefox' and 'chatgpt install'":
    #         print(f"{answer}\n"
    #               f"Can continue with {summary_pair.bug.id}: {summary_pair.rm_summary.id} {summary_pair.add_summary.id}")
    #         return True
    #     return False

    @staticmethod
    @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
    def ask_turbo(messages, model=TURBO_MODEL_NAME, temperature=1):
        """
        ask LLM question
        Args:
            temperature (): number Optional Defaults to 1
                            What sampling temperature to use, between 0 and 2.
                            Higher values like 0.8 will make the output more random,
                            while lower values like 0.2 will make it more focused and deterministic.
                            We generally recommend altering this or top_p but not both.
            model ():
            messages ():

        Returns: answer

        """
        # time.sleep(25)

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            # max_tokens=10240,
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer

    @staticmethod
    def get_messages_for_turbo(session_prompt, qa_pairs=None):
        """
        model="gpt-3.5-turbo",
        Args:
            system_role: for session_prompt,
            question_role: for question,
            answer_role: for answer,
            session_prompt: for system_role introduction
            qa_pairs (examples): (Q, A) pairs

        Returns:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        """
        messages = [{'role': LLMUtil.ROLE_SYSTEM, 'content': session_prompt}]
        if qa_pairs:
            for qa in qa_pairs:
                role_content_dict = {'role': LLMUtil.ROLE_USER, 'content': qa[0]}
                messages.append(role_content_dict)
                role_content_dict = {'role': LLMUtil.ROLE_ASSISTANT, 'content': qa[1]}
                messages.append(role_content_dict)
        return messages

    @staticmethod
    def add_role_content_dict_into_messages(role, content, messages):
        role_content_dict = {'role': role, 'content': content}
        messages.append(role_content_dict)
        return messages

    @staticmethod
    def show_messages(messages):
        for message in messages:
            if isinstance(message['content'], dict):
                pass
            else:
                print(f"{message['role']}: {message['content']}")
