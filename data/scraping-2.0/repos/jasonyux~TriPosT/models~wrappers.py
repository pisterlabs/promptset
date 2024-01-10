from models.base import GenerativeModel
from models.api_models import OpenAIWrapper, OpenAIChatWrapper
from utils.utils import format_multiple_choice
from langchain import LLMChain, PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

from prompts.word_sorting import *
from prompts.date_understanding import *
from prompts.multistep_arithmetic import *
from prompts.logical_deduction import *
from typing import Any, List, Mapping, Optional

import numpy as np
import re
import transformers


class HF_LLM_Wrapper(LLM):
    model_name_or_path: str
    tokenizer: transformers.PreTrainedTokenizer = None
    model: transformers.PreTrainedModel = None
    gen_kwargs: Mapping[str, Any]
    
    @property
    def _llm_type(self) -> str:
        return "hf"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        if self.model is None:
            # initialize
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                padding=True, truncation=True, return_tensors="pt"
            )
            print(f'using {self.tokenizer.pad_token=}, {self.tokenizer.pad_token_id=}')
            print(f'{self.tokenizer.all_special_tokens=}')

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path
            )
            self.model = self.model.cuda()
        
        encoded_input = self.tokenizer.encode(prompt, return_tensors="pt")
        encoded_input = encoded_input.cuda()

        _gen_kwargs = {
            "max_new_tokens": 512,
            "num_beams": 1,
            "temperature": 1e-5,
            "do_sample": False,
            **self.gen_kwargs
        }
        output = self.model.generate(
            input_ids=encoded_input,
            **_gen_kwargs
        )
        trimmed_output = self.tokenizer.decode(output[0][len(encoded_input[0]):], skip_special_tokens=True)
        return trimmed_output


TEMPLATE_MCQ_INIT_ANSWER = """
{examples}
Q: {question}
Options:
{formatted_choices}
Answer: Let's think step by step.{additional_info}
""".strip()


class LLM_MultipleChoice(GenerativeModel):
    def __init__(
            self, 
            model_name='code-davinci-002',
            init_answer_examples='',
            additional_info='',
            verbose=False,
            **model_kwargs):
        self.model_name = model_name
        self.init_answer_examples = init_answer_examples

        if model_name in ['chatgpt']:
            self.llm = OpenAIChatWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)
        elif 'model_checkpoints' in model_name:
            self.llm = HF_LLM_Wrapper(model_name_or_path=model_name, gen_kwargs=model_kwargs)
            self.init_answer_examples = ''  # no need assuming model is trained
            print("WARNING: init_answer_examples is set to empty string for model_checkpoints")
        else:
            self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)

        # self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)
        self.additional_info = additional_info
        self.prompt_init_answer = PromptTemplate(
            input_variables=["examples", "question", "formatted_choices", 'additional_info'],
            template=TEMPLATE_MCQ_INIT_ANSWER
        )
        self.chain_get_init_answer = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_init_answer, 
            output_key="init_answer",
            verbose=verbose
        )
        return
    
    def prepare_input(self, data: dict):
        question = data['inputs'].replace('\n\n', '\n').strip()
        choices = data['multiple_choice_targets']
        formatted_choices = format_multiple_choice(choices)

        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "formatted_choices": formatted_choices,
            "additional_info": self.additional_info
        }
        return init_answer_data
    
    def __clean_init_answer(self, init_answer):
        if 'model_checkpoints' in self.model_name:
            all_ans_steps = init_answer.split('\n')
            out_ans = []
            for step in all_ans_steps:
                if '[END]' in step:
                    step = step.replace('[END]', '')
                    out_ans.append(step.strip())
                    break
                out_ans.append(step.strip())
            return '\n'.join(out_ans)
        else:
            return init_answer.strip()
    
    def generate(self, input_data: dict, **gen_kwargs):        # evaluate
        init_answer_data = self.prepare_input(input_data)
        out = self.chain_get_init_answer(init_answer_data)
        out['init_answer'] = self.__clean_init_answer(out['init_answer'])
        if gen_kwargs.get('out_dict', False):
            return out
        return out['init_answer']


TEMPLATE_QA_INIT_ANSWER = """
{examples}
Q: {question}
Answer: Let's think step by step.{additional_info}
""".strip()


class LLM_QA(GenerativeModel):
    def __init__(
            self, 
            model_name='code-davinci-002',
            init_answer_examples='',
            additional_info='',
            verbose=False,
            **model_kwargs):
        self.model_name = model_name
        self.init_answer_examples = init_answer_examples
        if model_name in ['chatgpt']:
            self.llm = OpenAIChatWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)
        elif 'model_checkpoints' in model_name:
            self.llm = HF_LLM_Wrapper(model_name_or_path=model_name, gen_kwargs=model_kwargs)
            self.init_answer_examples = ''  # no need assuming model is trained
            print("WARNING: init_answer_examples is set to empty string for model_checkpoints")
        else:
            self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)

        # self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)
        self.additional_info = additional_info
        self.prompt_init_answer = PromptTemplate(
            input_variables=["examples", "question", 'additional_info'],
            template=TEMPLATE_QA_INIT_ANSWER
        )
        self.chain_get_init_answer = LLMChain(
            llm=self.llm, 
            prompt=self.prompt_init_answer, 
            output_key="init_answer",
            verbose=verbose
        )
        return
    
    def prepare_input(self, data: dict):
        question = '\n'.join(data['inputs'].split('\n')[:-1])
        question = question.strip()
        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "additional_info": self.additional_info
        }
        return init_answer_data

    def __clean_init_answer(self, init_answer):
        if 'model_checkpoints' in self.model_name:
            all_ans_steps = init_answer.split('\n')
            out_ans = []
            for step in all_ans_steps:
                if '[END]' in step:
                    step = step.replace('[END]', '')
                    out_ans.append(step.strip())
                    break
                out_ans.append(step.strip())
            return '\n'.join(out_ans)
        else:
            return init_answer.strip()
    
    def generate(self, input_data: dict, **gen_kwargs):        # evaluate
        init_answer_data = self.prepare_input(input_data)
        out = self.chain_get_init_answer(init_answer_data)
        out['init_answer'] = self.__clean_init_answer(out['init_answer'])
        if gen_kwargs.get('out_dict', False):
            return out
        return out['init_answer']


class LLM_WordSorting(LLM_QA):
    def __init__(
            self, 
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_answer_examples=EXAMPLES_WORDSORT_RATIONALE,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        return
    
    def prepare_input(self, data: dict):
        question = data['question'].strip()
        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "additional_info": self.additional_info
        }
        return init_answer_data


class LLM_DateUnderstanding(LLM_MultipleChoice):
    def __init__(
            self, 
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_answer_examples=EXAMPLES_DATE_UNDERSTANDING_RATIONALE,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        return
    
    def prepare_input(self, data: dict):
        question = data['question'].strip()
        formatted_choices = data['formatted_choices'].strip()
        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "formatted_choices": formatted_choices,
            "additional_info": self.additional_info
        }
        return init_answer_data


class LLM_MultistepArithmetic(LLM_QA):
    def __init__(
            self, 
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        additional_info = (
            ' Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
            'So, remember to always compute the expressions inside parentheses or brackets first.'
        )
        super().__init__(
            model_name=model_name,
            init_answer_examples=EXAMPLES_MULTISTEP_ARITHMETIC_RATIONALE,  # only used for testing LLM
            additional_info=additional_info,
            verbose=verbose,
            **model_kwargs
        )
        return
    
    def prepare_input(self, data: dict):
        question = data['question'].strip()
        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "additional_info": self.additional_info
        }
        return init_answer_data


class LLM_LogicalDeduction(LLM_MultipleChoice):
    def __init__(
            self, 
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_answer_examples=EXAMPLES_LOGICAL_DEDUCTION_RATIONALE,
            verbose=verbose,
            additional_info=""" Let "??" represents 0 or more objects, and "?" represents exactly 1 object.""",
            **model_kwargs
        )
        return
    
    def prepare_input(self, data: dict):
        question = data['question'].strip()
        formatted_choices = data['formatted_choices'].strip()
        init_answer_data = {
            "examples": self.init_answer_examples,
            "question": question,
            "formatted_choices": formatted_choices,
            "additional_info": self.additional_info
        }
        return init_answer_data


TEMPLATE_QA_FEEDBACK = """
{examples}
Q: {question}
Answer: Let's think step by step.{additional_info}
{attempted_answer}
Feedback:
""".strip()

class LLM_QA_Feedback(GenerativeModel):
    def __init__(
            self, 
            model_name='code-davinci-002',
            init_feedback_examples='',
            additional_info='',
            verbose=False,
            **model_kwargs):
        self.model_name = model_name
        if model_name in ['chatgpt']:
            sys_messages = "You are a helpful teacher providing concise feedbacks to a student."
            self.llm = OpenAIChatWrapper(prefix_sys_messages=sys_messages, model_name=model_name, stop=['[END]'], **model_kwargs)
        elif 'model_checkpoints' in model_name:
            self.llm = HF_LLM_Wrapper(model_name_or_path=model_name, gen_kwargs=model_kwargs)
        else:
            self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)

        # Completion seems to be compatbile with chatgpt model, and it seems to perform better as well
        # self.llm = OpenAIWrapper(model_name=model_name, stop=['[END]'], **model_kwargs)
        self.init_feedback_examples = init_feedback_examples
        self.additional_info = additional_info

        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        self.verbose = verbose
        return
    
    def prepare_input(self, data: dict):
        question = data['question'].strip()
        attempted_answer = data['attempted_answer']
        init_feedback_data = {
            "examples": self.init_feedback_examples,
            "question": question,
            "attempted_answer": attempted_answer,
            "additional_info": self.additional_info
        }
        return init_feedback_data
    
    def __clean_init_answer(self, init_answer):
        if 'model_checkpoints' in self.model_name:
            all_ans_steps = init_answer.split('\n')
            out_ans = []
            for step in all_ans_steps:
                if '[END]' in step:
                    step = step.replace('[END]', '')
                    out_ans.append(step.strip())
                    break
                out_ans.append(step.strip())
            return '\n'.join(out_ans)
        else:
            return init_answer.strip()
    
    def generate(self, input_data: dict, **gen_kwargs):        # evaluate
        init_feedback_data = self.prepare_input(input_data)
        out = self.chain_get_feedback(init_feedback_data)
        out['feedback'] = self.__clean_init_answer(out['feedback'])
        if gen_kwargs.get('out_dict', False):
            return out
        return out['feedback']


TEMPLATE_QA_FEEDBACK_TABULAR = """
{examples}
Q: {question}
Answer: Let's think step by step.{additional_info}
{attempted_answer}
Earliest error step:
""".strip()


class LLM_Feedback_NoCorrect_Tabular(LLM_QA_Feedback):    
    def _format_feedback(self, raw_generated, attempted_num_steps:int):
        if len(raw_generated.split('\n')) < 3:
            return '[ERROR] ' + raw_generated
        lines = raw_generated.split('\n')[:3]
        # parse error step
        error_steps = re.findall(r'\((\d+)\)', lines[0].strip())
        if len(error_steps) == 0 and 'final response' not in lines[0].lower():
            if self.verbose:
                print("No error steps found:", lines[0].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        try:
            if 'final response' not in lines[0].lower():
                error_step = int(error_steps[0])
                if error_step > attempted_num_steps - 1:
                    if self.verbose:
                        print("Error step larger than attempted steps:", lines[0].strip())
                        print("Raw generated:", raw_generated)
                    return '[ERROR] ' + raw_generated
            else:
                error_step = attempted_num_steps
        except:
            if self.verbose:
                print("Error step not integer:", lines[0].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated

        # parse error segment
        if "Error segment:" not in lines[1]:
            if self.verbose:
                print("No error segment found:", lines[1].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_segment = lines[1].replace("Error segment:", "").strip()
        if not error_segment.startswith('"'):
            error_segment = '"' + error_segment
        if not error_segment.endswith('"'):
            error_segment = error_segment + '"'
        
        # parse error reason
        if "Error reason:" not in lines[2]:
            if self.verbose:
                print("No error reason found:", lines[2].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_reason = lines[2].replace("Error reason:", "").strip()
        if not error_reason.endswith('.'):
            error_reason = error_reason + '.'
        error_reason = error_reason[0].lower() + error_reason[1:]

        # format
        if error_step == 1:
            formatted_feedback = f"""
            In step (1) the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        elif error_step == 2:
            error_step_str = str(error_step)
            if error_step == attempted_num_steps:
                error_step_str = "Final response"
            formatted_feedback = f"""
            Step (1) is correct. In step ({error_step_str}) the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        else:
            error_step_str = str(error_step)
            if error_step == attempted_num_steps:
                error_step_str = "Final response"
            formatted_feedback = f"""
            Step (1) to step ({error_step-1}) are correct. In step ({error_step_str}) the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        return formatted_feedback.strip()
    
    def __clean_init_answer(self, init_answer):
        if 'model_checkpoints' in self.model_name:
            all_ans_steps = init_answer.split('\n')
            out_ans = []
            for step in all_ans_steps:
                if '[END]' in step:
                    step = step.replace('[END]', '')
                    out_ans.append(step.strip())
                    break
                out_ans.append(step.strip())
            return '\n'.join(out_ans)
        else:
            return init_answer.strip()

    def generate(self, input_data: dict, **gen_kwargs):        # evaluate
        init_feedback_data = self.prepare_input(input_data)
        out = self.chain_get_feedback(init_feedback_data)

        attempted_num_steps = len(input_data['attempted_answer'].split('\n'))
        out['feedback'] = self.__clean_init_answer(out['feedback']).strip()
        out['feedback'] = self._format_feedback(out['feedback'], attempted_num_steps)
        if gen_kwargs.get('out_dict', False):
            return out
        return out['feedback']


class LLM_WordSorting_Feedback_NoCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_WORDSORT_FEEDBACK_NOCORRECT_TABULAR,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return
    
    def _format_feedback(self, raw_generated, attempted_num_steps:int):
        lines = raw_generated.split('\n')[:3]
        # parse error step
        error_steps = re.search(r"^\((\d+)(\.\d+)*\)", lines[0].strip())
        if 'final response' in lines[0].lower():
            error_step = '(Final response)'
        elif error_steps is None:
            if self.verbose:
                print("No error steps found:", lines[0].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        else:
            error_step = error_steps.group(0).strip()

        # parse error segment
        if "Error segment:" not in lines[1]:
            if self.verbose:
                print("No error segment found:", lines[1].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_segment = lines[1].replace("Error segment:", "").strip()
        if not error_segment.startswith('"'):
            error_segment = '"' + error_segment
        if not error_segment.endswith('"'):
            error_segment = error_segment + '"'
        
        # parse error reason
        if "Error reason:" not in lines[2]:
            if self.verbose:
                print("No error reason found:", lines[2].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_reason = lines[2].replace("Error reason:", "").strip()
        if not error_reason.endswith('.'):
            error_reason = error_reason + '.'
        error_reason = error_reason[0].lower() + error_reason[1:]

        # format
        if error_step == '(1)':
            formatted_feedback = f"""
            In step (1) the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        elif error_step == '(2)':
            formatted_feedback = f"""
            Step (1) is correct. In step {error_step} the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        else:
            formatted_feedback = f"""
            Step (1) until step {error_step} are correct. In step {error_step} the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        return formatted_feedback.strip()


class LLM_WordSorting_Feedback_wCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_WORDSORT_FEEDBACK_HASCORRECT_TABULAR,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return
    
    def _format_feedback(self, raw_generated, attempted_num_steps:int):
        lines = raw_generated.split('\n')[:3]
        # parse error step
        error_steps = re.search(r"^\((\d+)(\.\d+)*\)", lines[0].strip())
        if 'final response' in lines[0].lower():
            error_step = '(Final response)'
        elif error_steps is None:
            if self.verbose:
                print("No error steps found:", lines[0].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        else:
            error_step = error_steps.group(0).strip()

        # parse error segment
        if "Error segment:" not in lines[1]:
            if self.verbose:
                print("No error segment found:", lines[1].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_segment = lines[1].replace("Error segment:", "").strip()
        if not error_segment.startswith('"'):
            error_segment = '"' + error_segment
        if not error_segment.endswith('"'):
            error_segment = error_segment + '"'
        
        # parse error reason
        if "Error reason:" not in lines[2]:
            if self.verbose:
                print("No error reason found:", lines[2].strip())
                print("Raw generated:", raw_generated)
            return '[ERROR] ' + raw_generated
        error_reason = lines[2].replace("Error reason:", "").strip()
        if not error_reason.endswith('.'):
            error_reason = error_reason + '.'
        error_reason = error_reason[0].lower() + error_reason[1:]

        # format
        if error_step == '(1)':
            formatted_feedback = f"""
            In step (1) the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        elif error_step == '(2)':
            formatted_feedback = f"""
            Step (1) is correct. In step {error_step} the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        else:
            formatted_feedback = f"""
            Step (1) until step {error_step} are correct. In step {error_step} the part {error_segment} is incorrect. This is because {error_reason}
            """.replace("    ", "")
        return formatted_feedback.strip()


class LLM_DateUnderstanding_Feedback_NoCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_DATE_UNDERSTANDING_FEEDBACK_NOCORRECT_TABULAR,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return
    

class LLM_DateUnderstanding_Feedback_wCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_DATE_UNDERSTANDING_FEEDBACK_HASCORRECT_TABULAR,
            additional_info='',
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return


class LLM_MultistepArithmetic_Feedback_NoCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        additional_info = (
            ' Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
            'So, remember to always compute the expressions inside parentheses or brackets first.'
        )
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_MULTISTEP_ARITHMETIC_FEEDBACK_NOCORRECT_TABULAR,
            additional_info=additional_info,
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return
    

class LLM_MultistepArithmetic_Feedback_wCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        additional_info = (
            ' Recall that the order of operations in mathematics is as follows: (1) Parentheses, (2) exponents, (3) multiplication and division (from left to right), (4) addition and multiplication (from left to right). '
            'So, remember to always compute the expressions inside parentheses or brackets first.'
        )
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_MULTISTEP_ARITHMETIC_FEEDBACK_HASCORRECT_TABULAR,
            additional_info=additional_info,
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return


class LLM_LogicalDeduction_Feedback_NoCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_LOGICAL_DEDUCTION_FEEDBACK_NOCORRECT_TABULAR,
            additional_info=""" Let "??" represents 0 or more objects, and "?" represents exactly 1 object.""",
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return


class LLM_LogicalDeduction_Feedback_wCorrect_Tabular(LLM_Feedback_NoCorrect_Tabular):
    def __init__(
            self,
            model_name='code-davinci-002',
            verbose=False,
            **model_kwargs):
        super().__init__(
            model_name=model_name,
            init_feedback_examples=EXAMPLES_LOGICAL_DEDUCTION_FEEDBACK_HASCORRECT_TABULAR,
            additional_info=""" Let "??" represents 0 or more objects, and "?" represents exactly 1 object.""",
            verbose=verbose,
            **model_kwargs
        )
        self.verbose = verbose
        self.prompt_feedback = PromptTemplate(
            input_variables=["examples", "question", "attempted_answer", "additional_info"],
            template=TEMPLATE_QA_FEEDBACK_TABULAR
        )
        self.chain_get_feedback = LLMChain(
            llm=self.llm,
            prompt=self.prompt_feedback,
            output_key="feedback",
            verbose=verbose
        )
        return


TEMPLATE_MODEL_QA_INIT_ANSWER = """
Q: {question}
Answer:{additional_info}
""".strip()


class GPT_QA(GenerativeModel):
    def __init__(self, 
            model, tokenizer, 
            additional_info='',  # if empty, same as answer only. 
            input_max_length=1024, max_new_tokens=1024,
            gen_kwargs={}):
        """used for baseline, directly generating a rationale or answer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.additional_info = additional_info
        self.input_max_length = input_max_length
        self.max_new_tokens = max_new_tokens
        self.gen_kwargs = gen_kwargs
        return

    def _extract_final_answer(self, answer_line):
        return answer_line

    def _format_task_output(self, output_text):
        output_text = output_text.strip()
        # remove new questions it generated
        steps = output_text.split("\n")
        cleaned_steps = [steps[0]]
        for step in steps[1:]:
            # started a new question by itself
            # llama generates this at the end sometimes
            step = step.replace("⁇", "").strip()
            if 'Q:' in step:
                break
            cleaned_steps.append(step)
            if '(Final response)' in step:
                break
        cleaned_response = "\n".join(cleaned_steps)
        start_idx = cleaned_response.find("Answer:")
        if start_idx == -1:
            return ''
        cleaned_answer = cleaned_response[start_idx:]

        # if it is step by step, we are done because evaluator looks for "the answer is:" in the last step
        if "Let's think step by step".lower() in cleaned_response.lower():
            return cleaned_answer
        
        # otherwise, we help a bit by directly spit out answer
        answer_line = ""
        for step in steps[1:]:
            if 'Answer:' in step:
                answer_line = step
                break
        final_answer = self._extract_final_answer(answer_line)
        return final_answer
    
    def _prepare_input(self, input_data: dict):
        raise NotImplementedError
    
    def generate(self, input_data: dict, **gen_kwargs):
        if 'batched_input' in input_data:
            input_text = []
            for input_data_ in input_data['batched_input']:
                input_text.append(self._prepare_input(input_data_))
        else:
            input_text = self._prepare_input(input_data)
        
        self.tokenizer.truncation_side = "left"  # truncate old info for prediction
        self.tokenizer.padding_side = "left"
        encoded_input = self.tokenizer(
            input_text, 
            max_length=self.input_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        if 'token_type_ids' in encoded_input:
            encoded_input.pop('token_type_ids')
        
        # encoded_input = self._fixed_padding(encoded_input)  # when we pad left with eos, this is needed
        for k in encoded_input:
            encoded_input[k] = encoded_input[k].to(self.model.device)

        all_gen_kwargs = {
            **self.gen_kwargs,
            **gen_kwargs
        }
        model_output = self.model.generate(
            **encoded_input,
            **all_gen_kwargs,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            early_stopping=True,
        )
        decoded_model_output = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        if 'batched_input' in input_data:
            cleaned_model_output = []
            for decoded_model_output_ in decoded_model_output:
                cleaned_model_output.append(self._format_task_output(decoded_model_output_))
        else:
            cleaned_model_output = self._format_task_output(decoded_model_output[0])
        return cleaned_model_output


class GPT_QA_PseudoSelfImprove(GPT_QA):
    def _format_task_output(self, output_text):
        output_text = output_text.strip()
        # remove new questions it generated
        steps = output_text.split("\n")
        cleaned_steps = [steps[0]]
        for step in steps[1:]:
            # started a new question by itself
            # llama generates this at the end sometimes
            step = step.replace("⁇", "").strip()
            cleaned_steps.append(step)
            if 'Q:' in step:
                break
        cleaned_response = "\n".join(cleaned_steps)
        # if it is step by step, we are done because evaluator looks for "the answer is:" in the last step
        if "Let's think step by step".lower() not in cleaned_response.lower():
            return ''
        return cleaned_response.strip()


class GPT_WordSorting(GPT_QA):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale or answer
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return
    
    def _extract_final_answer(self, answer_line):
        if answer_line == "":
            return []
        else:
            cleaned_answer = answer_line.replace("Answer:", "").strip().split()
        return cleaned_answer
    
    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        input_text = TEMPLATE_MODEL_QA_INIT_ANSWER.format(question=question, additional_info=self.additional_info)
        return input_text
    

class GPT_WordSort_PseudoSelfImprove(GPT_QA_PseudoSelfImprove):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale and convert it to a self-improve like response
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return

    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        input_text = TEMPLATE_MODEL_QA_INIT_ANSWER.format(question=question, additional_info=self.additional_info)
        return input_text


class GPT_MultistepArithmetic(GPT_QA):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale or answer
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return
    
    def _extract_final_answer(self, answer_line):
        if answer_line == "":
            return np.inf
        try:
            cleaned_answer_text = re.findall(r'\d*\.?\d+', answer_line)[0]
            cleaned_answer = float(cleaned_answer_text)
        except:
            cleaned_answer = np.inf
        return cleaned_answer
    
    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        input_text = TEMPLATE_MODEL_QA_INIT_ANSWER.format(question=question, additional_info=self.additional_info)
        return input_text
    

class GPT_MultistepArithmetic_PseudoSelfImprove(GPT_QA_PseudoSelfImprove):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale and convert it to a self-improve like response
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return

    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        input_text = TEMPLATE_MODEL_QA_INIT_ANSWER.format(question=question, additional_info=self.additional_info)
        return input_text


TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER = """
Q: {question}
Options:
{formatted_choices}
Answer:{additional_info}
""".strip()


class GPT_DateUnderstanding(GPT_QA):
    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        formatted_choices = input_data['formatted_choices'].strip()
        input_text = TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER.format(
            question=question,
            formatted_choices=formatted_choices,
            additional_info=self.additional_info
        )
        return input_text


class GPT_DateUnderstanding_PseudoSelfImprove(GPT_QA_PseudoSelfImprove):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale and convert it to a self-improve like response
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return

    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        formatted_choices = input_data['formatted_choices'].strip()
        input_text = TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER.format(
            question=question,
            formatted_choices=formatted_choices,
            additional_info=self.additional_info
        )
        return input_text
    

class GPT_LogicalDeduction(GPT_QA):
    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        formatted_choices = input_data['formatted_choices'].strip()
        input_text = TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER.format(
            question=question,
            formatted_choices=formatted_choices,
            additional_info=self.additional_info
        )
        return input_text


class GPT_LogicalDeduction_PseudoSelfImprove(GPT_QA_PseudoSelfImprove):
    def __init__(self, model, tokenizer, additional_info, input_max_length=1024, max_new_tokens=1024, gen_kwargs={}):
        """used for baseline, directly generating a rationale and convert it to a self-improve like response
        """
        super().__init__(
            model=model, tokenizer=tokenizer,
            additional_info=additional_info,  # if empty, same as answer only.
            input_max_length=input_max_length, max_new_tokens=max_new_tokens,
            gen_kwargs=gen_kwargs
        )
        return

    def _prepare_input(self, input_data: dict):
        question = input_data['question'].strip()
        formatted_choices = input_data['formatted_choices'].strip()
        input_text = TEMPLATE_MODEL_MULTICHOICE_INIT_ANSWER.format(
            question=question,
            formatted_choices=formatted_choices,
            additional_info=self.additional_info
        )
        return input_text