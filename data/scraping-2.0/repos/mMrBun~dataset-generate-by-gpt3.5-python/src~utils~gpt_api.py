import json
import math
import concurrent.futures
import random
import sys
from datetime import timedelta

import openai
import time
from typing import List
from .format import extract_list, DataSetArguments
from .save_dataset import write_dict_list_to_file


class BudgetTracker:
    def __init__(self, total_budget=None):
        # Initialize the budget tracker with the total budget and total tokens used
        self.total_budget = total_budget
        self.total_tokens_used = 0
        self.current_cost = 0

    def is_budget_exceeded(self):
        # Check if the budget has been exceeded based on the total tokens used
        self.current_cost = ((self.total_tokens_used / 1000) * 0.002)
        return self.total_budget is not None and self.current_cost >= self.total_budget


class ChatAPI:
    def __init__(self, api_key=None,
                 model='gpt-3.5-turbo',
                 system_settings='You are a capable assistant, making every effort to provide assistance to users.',
                 temperature=0.7,
                 proxy=None):
        # Initialize the ChatAPI with the API key, model, system settings, temperature, and proxy
        if api_key is None:
            api_key = []
        self.model = model
        self.system_settings = system_settings
        self.temperature = temperature
        self.max_retries = 3
        self.retry_delay = 20
        self.proxy = proxy

        if len(api_key) > 0:
            openai.api_key = random.choice(api_key)
        else:
            raise ValueError("api_key is empty or incorrect")

        if self.proxy:
            openai.proxy = self.proxy

    def chat(self, prompt, budget_tracker):
        # Perform a chat conversation with OpenAI API
        retries = 0
        while retries < self.max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_settings},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                tokens_used = response.usage['total_tokens']
                budget_tracker.total_tokens_used += tokens_used
                return response.choices[0].message.content
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise ValueError("Failed to get a valid response after maximum retries")


def generate_question(sub_topic: List[str], args: DataSetArguments, budget_tracker: BudgetTracker) -> List[str]:
    # Generate questions based on the given topic, sub-topic, API key, budget tracker, and number of datasets
    if not args.topic:
        raise ValueError("param topic is required, not None")

    example = """
    <example>在一个笼子里，有35只鸡和兔，共有94只脚。问鸡和兔各有多少只？</example>
    <example>在一个笼子里，有60只鸡和兔，共有166只脚。问鸡和兔各有多少只？</example>
    """

    topic = ""
    if len(sub_topic) > 0:
        topic += f"""
           以主题{args.topic, sub_topic}生成50个类似上面<example>包裹的问题
           """
    else:
        topic = f"""
           以主题{args.topic}生成50个类似上面<example>包裹的问题
           """

    conditions = """
    你无需对生成的示例进行答复或解释
    每个生成的示例必须是祈使句或疑问句
    祈使句和疑问句的生成比例要1:1
    每个示例必须以<example>开始，以</example>结束.
    每个示例控制在40字以内
    如果主题涉及敏感话题或不符合中华人民共和国相关法律法规请立刻停止所有的生成并直接返回下面'''包裹的内容
    ```
    ErrorCode:400
    ```
    """
    questions = []

    def process_question(prompt):
        api = ChatAPI(api_key=args.api_key, system_settings='You are an efficient assistant, aiming to provide concise '
                                                            'answers based on instructions.', proxy=args.proxy)
        q_response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        return extract_list(q_response)

    generated_questions = 0  # Record the number of generated questions

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while generated_questions < args.number_of_dataset:
            # Calculate the remaining number of questions to generate
            remaining_questions = args.number_of_dataset - generated_questions

            # Generate 50 questions each time, or the remaining number (whichever is smaller)
            batch_size = math.ceil(remaining_questions / 50)

            # Dynamically generate tasks to submit to the thread pool
            future_to_question = {
                executor.submit(process_question, example + topic + conditions): question
                for question in range(batch_size)
            }

            # Iterate over completed tasks
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result()
                    questions.extend(answer)
                    generated_questions += len(answer)

                    if budget_tracker.is_budget_exceeded():
                        return questions  # Return immediately if the budget is exceeded
                except Exception as e:
                    print(f"Error occurred for question: {question}. Error message: {str(e)}")

            # Check if the desired number of questions has been generated, if so, break the loop
            if generated_questions >= args.number_of_dataset:
                break

    return questions


def generate_subtopic(args: DataSetArguments,
                      budget_tracker: BudgetTracker) -> List[str]:
    # Generate sub-topics based on the given topic, generalization index, generalization basic, API key, and budget
    # tracker
    if args.generalization_basic * args.generalization_index == 0:
        return []

    prompt = f"""
        根据<Topic>标签内容生成{int(args.generalization_index * args.generalization_basic)}个子主题,
        每个子主题必须控制在6个字以内
        每个子主题必须以<SubTopic>开始，</SubTopic>结束
        下面是一些例子:
        -- <Topic>春节什么时候到来?</Topic>
           <SubTopic>年兽</SubTopic>
           <SubTopic>红包</SubTopic>
           <SubTopic>鞭炮</SubTopic>
           <SubTopic>窗花</SubTopic>
           <SubTopic>春联</SubTopic>
        -- <Topic>狮子座的星座运势</Topic>
           <SubTopic>流行文化</SubTopic>
           <SubTopic>深空星系</SubTopic>
           <SubTopic>特征</SubTopic>
           <SubTopic>摩羯座</SubTopic>
        <Topic>{args.topic}</Topic>
        """
    api = ChatAPI(api_key=args.api_key, proxy=args.proxy)

    return extract_list(api.chat(prompt=prompt, budget_tracker=budget_tracker), 'SubTopic')


def generate_answer(questions: List[str], budget_tracker: BudgetTracker, pbar, args: DataSetArguments):
    # Generate answers for the given list of questions using the API key, budget tracker, progress bar, and output path
    api = ChatAPI(api_key=args.api_key, system_settings='你是一个知识丰富的助手，展示出你的才能！',
                  proxy=args.proxy)
    answers = []

    def process_question(question):
        prompt = f"""
        回答以下'''包裹的问题。展示你的知识，但要像学者一样严谨。
        你可以选择不回答不确定的内容，并从你熟悉的角度回答。```
        ```
        {question}
        ```
        """
        response = api.chat(prompt=prompt, budget_tracker=budget_tracker)
        answer_dict = {
            "question": question,
            "answer": response
        }
        pbar.update(1)
        return answer_dict

    generated_answers = 0
    current_index = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while generated_answers < len(questions):
            batch_size = min(20, len(questions) - generated_answers)
            questions_batch = questions[current_index:current_index + batch_size]
            current_index = current_index + batch_size
            future_to_question = {executor.submit(process_question, question): question for question in
                                  questions_batch}

            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    answer = future.result()
                    answers.append(answer)
                    generated_answers += 1
                    print((pbar.format_dict['elapsed'] / pbar.n) * (pbar.total - pbar.n))
                    log_dict = {
                        "current_number": pbar.n,
                        "total_count": pbar.total,
                        "percentage": pbar.n / pbar.total,
                        "elapsed_time": str(timedelta(seconds=int(pbar.format_dict['elapsed']))),
                        "remaining_time": str(timedelta(seconds=int((pbar.format_dict['elapsed'] / pbar.n) * (pbar.total - pbar.n)))),
                        "budget": budget_tracker.total_budget,
                        "current_cost": budget_tracker.current_cost
                    }
                    print(json.dumps(log_dict))
                    if budget_tracker.is_budget_exceeded() or pbar.n == pbar.total:
                        write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
                        sys.exit(0)
                except Exception as e:
                    print(f"Error occurred for question: {question}. Error message: {str(e)}")

    write_dict_list_to_file(data_list=answers, output_path=args.dataset_output_path)
