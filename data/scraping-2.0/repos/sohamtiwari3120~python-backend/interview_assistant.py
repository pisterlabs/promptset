import argparse
from dotenv import dotenv_values
from typing import Optional
import json
from dotenv import dotenv_values
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.runnable import ConfigurableField
from langchain.chat_models import ChatOpenAI
from instructions import (
    generic_instruct_ctx,
    bug_instruct_ctx,
    bug_instruct_cot,
    mode_instruct_ctx,
    concept_instruct_w_code,
    concept_instruct_ctx,
    concept_instruct,
    direct_q,
    direct_q_retrieval,
    intro,
)
from general_utils import get_retrieval_index
from langchain.vectorstores import Pinecone


class InterviewAssistant:
    def __init__(
        self,
        coding_question: str,
        code_solution: str,
        api_key: str,
        initial_mode: str = "conceptual",
        mode_switching: str = "heuristic",
        include_retrieved_ctx: bool = True,
        heuristic_switchover=None,
        max_token_args=None,
        retrieval_idx: str = "capstone-langchain-retrieval-augmentation2",
        in_context_examples: int = 3,
        cot: bool = False
    ):
        self.api_key = api_key
        self.mode = initial_mode
        self.coding_q = coding_question
        self.solution = code_solution
        self.mode_switching = mode_switching
        self.num_invocations = 0
        self.temperature = 0.8
        self.model = "gpt-3.5-turbo-1106"
        self.chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=250,
            api_key=self.api_key,
        ).configurable_fields(
            max_tokens=ConfigurableField(
                id="max_tokens",
                name="Max Tokens",
                description="Maximum Tokens to output",
            )
        )
        self.include_retrieved_ctx = include_retrieved_ctx
        if self.include_retrieved_ctx: 
            embed_model_name = 'text-embedding-ada-002'
            self.embed = OpenAIEmbeddings(
                model=embed_model_name,
                openai_api_key=self.api_key)
            index = get_retrieval_index(retrieval_idx)
            vectorstore = Pinecone(
                index, self.embed.embed_query, "text"
            )
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

        if heuristic_switchover is None:
            self.heuristic_switchover = 2
        else:
            self.heuristic_switchover = heuristic_switchover
        if (
            max_token_args is None
            or "hint_max" not in max_token_args
            or "direct_question_response_max" not in max_token_args
        ):
            self.max_token_args = {"hint_max": 250, "direct_question_response_max": 500}
        else:
            self.max_token_args = max_token_args
        self.in_context_examples = in_context_examples
        self.cot = cot
        # Stop words
        self.stop_words = [
            "Question:",
            "Current Code:",
            "Current Transcript:",
            "Solution:",
        ]

    def __call__(
        self,
        current_code: str,
        current_transcript: str,
        question: Optional[str] = None,
        direct_question_flg: bool = False,
        in_context: bool = False
    ):
        if direct_question_flg:
            if question is None:
                raise ValueError(
                    "No explicit Question has Been Passed in " "But Direct Flag is here"
                )
            return self.direct_question_response(
                current_code, current_transcript, question
            )
        self.num_invocations += 1
        # Elicit either a conceptual, high-level hint or a fine-grained hint
        if self.mode != "generic":
            self.mode = self.determine_mode(current_code, current_transcript)
            print(f"Current Mode: {self.mode}")
        if self.mode == "conceptual":
            instruction = concept_instruct
            if current_code: 
                instruction = concept_instruct_w_code
        elif self.mode == "fine-grained":
            if in_context: 
                instruction = self.in_context_prompt()
            else: 
                if self.cot: 
                    instruction = bug_instruct_cot
                else: 
                    instruction = bug_instruct_ctx
        else:
            instruction = generic_instruct_ctx
        # print(f"Instruction: {instruction}")
        return self.generate_chat_response(
            instruction,
            coding_question=self.coding_q,
            code_snippet=current_code,
            interview_transcript=current_transcript,
            solution=self.solution,
            max_tokens=self.max_token_args["hint_max"],
        )

    def generate_chat_response(
        self,
        sys_instruction: str,
        coding_question: str,
        code_snippet: str,
        interview_transcript: str,
        solution: str = "",
        max_tokens: int = 250,
    ):
        # Construct user prompt
        user_prompt = f"Question: {coding_question}\nCurrent Transcript: {interview_transcript}"
        if code_snippet:
            user_prompt += f"\nStudent Code: {code_snippet}"
        if solution and self.mode == "fine-grained":
            user_prompt += f"\nSolution: {solution}"
        messages = [
            SystemMessage(content=sys_instruction),
            HumanMessage(content=user_prompt),
        ]
        ai_response = self.chat.with_config(
            configurable={"max_tokens": max_tokens}
        ).invoke(messages, stop=self.stop_words)
        response = ai_response.content
        if self.cot: 
            hint_start = response.find("Hint:")
            response = response[hint_start + 6:]
        return response

    def determine_mode(self, code, transcript):
        if self.mode_switching == "heuristic":
            if self.num_invocations < self.heuristic_switchover or not code:
                return "conceptual"
            else:
                return "fine-grained"
        else:
            llm_response = self.generate_chat_response(
                mode_instruct_ctx,
                coding_question=self.coding_q,
                code_snippet=code,
                interview_transcript=transcript,
                solution=self.solution,
                max_tokens=250,
            )
    
            if "yes" in llm_response.lower() or not code:
                return "conceptual"
            else:
                return "fine-grained"

    def direct_question_response(self, code, transcript, question) -> str:
        # Construct user prompt
        user_prompt = f"Question: {self.coding_q}\nStudent Code: {code}\nCurrent Transcript: {transcript}"
        user_prompt += f"\nSolution: {self.solution}"
        instruct = direct_q
        if self.include_retrieved_ctx:
            retrieved_docs = self.retriever.invoke(question)
            retrieved_ctx = retrieved_docs[0].page_content
            user_prompt += f"\nContext: {retrieved_ctx}"
            # print(f"Retrieved Context: {retrieved_ctx}")
            instruct = direct_q_retrieval
        user_prompt  += f"\nStudent Question: {question}"

        messages = [SystemMessage(content=instruct), HumanMessage(content=user_prompt)]
        ai_response = self.chat.with_config(
            configurable={
                "max_tokens": self.max_token_args["direct_question_response_max"]
            }
        ).invoke(messages, stop=self.stop_words)
        return ai_response.content
    
    def in_context_prompt(self):
        instruction = ""
        if self.cot: 
            example_file = 'train_cot'
            instruction = bug_instruct_cot
        else: 
            example_file = 'train'
            instruction = bug_instruct_ctx
        examples_added = 0
        with open(f"{example_file}.jsonl", "r") as f:
            for example in f:
                if examples_added == self.in_context_examples: 
                    break
                current_example = json.loads(example)
                messages = current_example["messages"]
                instruction += messages[1]["content"]
                instruction += "\n Assistant: "
                instruction += messages[2]["content"]
                instruction += "\n"
                examples_added += 1
        return instruction
        

    def check_hint(self, hint, true_ans) -> bool:
        raise NotImplementedError

    def upgrade_model(self):
        self.model = "gpt-4-1106-preview"
        self.chat = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=250,
            api_key=self.api_key,
        ).configurable_fields(
            max_tokens=ConfigurableField(
                id="max_tokens",
                name="Max Tokens",
                description="Maximum Tokens to output",
            )
        )
        self.mode = "generic"

    def override_mode(self, mode):
        self.mode = mode

    def get_current_mode(self):
        return self.mode

if __name__ == "__main__":
    all_user_prompts = []
    cot = False
    if cot: 
        example_file = 'train_cot'  
    else: 
        example_file = 'train'

    with open(f"{example_file}.jsonl", "r") as f:
        for example in f:
            current_ex = json.loads(example)
            messages = current_ex["messages"]
            all_user_prompts.append(messages[1]["content"])
    chosen_example_idx = 5
    example = all_user_prompts[chosen_example_idx]

    transcript_start_idx = example.find("Current Transcript:")
    student_code_start_idx = example.find("Student Code:")
    solution_start_idx = example.find("Solution:")

    question = example[10:transcript_start_idx]
    solution = example[solution_start_idx+10:]
    transcript = example[transcript_start_idx+19:student_code_start_idx]
    student_code = example[student_code_start_idx+14:solution_start_idx]

    config = dotenv_values(".env")
    OPENAI_KEY = config["OPENAI_KEY"]
    interview_agent = InterviewAssistant(
        coding_question=question,
        code_solution=solution,
        api_key=OPENAI_KEY,
        initial_mode="fine-grained",
        mode_switching="model",
        cot=True
    )
    print(f"Question: {question}")
    print(f"Response: {interview_agent(student_code, transcript)}")
    print(f"Response w/ ICT: {interview_agent(student_code, transcript, in_context=True)}")

