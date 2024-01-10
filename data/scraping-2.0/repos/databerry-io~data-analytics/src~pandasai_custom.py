
# import ast
import io
# import logging
import re
# import sys
import uuid
import time
from contextlib import redirect_stdout
from typing import List, Optional, Union
# from pandasai.middlewares.streamlit import StreamlitMiddleware

# import astor
import pandas as pd

# Configure it such that pandas head does NOT truncate the dataframe
pd.set_option('display.max_columns', None)

# from pandasai.constants import (
#     WHITELISTED_BUILTINS,
#     WHITELISTED_LIBRARIES,
# )
# from pandasai.exceptions import BadImportError, LLMNotFoundError
# from pandasai.helpers._optional import import_dependency
from pandasai.helpers.anonymizer import anonymize_dataframe_head
# from pandasai.helpers.cache import Cache
# from pandasai.helpers.notebook import Notebook
from pandasai.helpers.save_chart import add_save_chart
# from pandasai.helpers.shortcuts import Shortcuts
# from pandasai.llm.base import LLM
# from pandasai.llm.langchain import LangchainLLM
# from pandasai.middlewares.base import Middleware
# from pandasai.middlewares.charts import ChartsMiddleware
from pandasai.prompts.correct_error_prompt import CorrectErrorPrompt
from pandasai.prompts.correct_multiples_prompt import CorrectMultipleDataframesErrorPrompt
from pandasai.prompts.generate_python_code import GeneratePythonCodePrompt
# from pandasai.prompts.generate_response import GenerateResponsePrompt
from pandasai.prompts.multiple_dataframes import MultipleDataframesPrompt

from pandasai import PandasAI
import contextlib


from .prompts import CodeSummaryPrompt, ColumnKeyErrorPrompt, GraphCleaupPrompt

class ExceededMaxRetriesError(Exception):
    """Raised when the maximum number of retries is exceeded"""

class CustomPandasAI(PandasAI):
    def run(
        self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
        df_head: Optional[pd.DataFrame] = None,
    ) -> Union[str, pd.DataFrame]:
        """
        Run the PandasAI to make Dataframes Conversational.

        Args:
            data_frame (Union[pd.DataFrame, List[pd.DataFrame]]): A pandas Dataframe
            prompt (str): A prompt to query about the Dataframe
            is_conversational_answer (bool): Whether to return answer in conversational
            form. Default to False
            show_code (bool): To show the intermediate python code generated on the
            prompt. Default to False
            anonymize_df (bool): Running the code with Sensitive Data. Default to True
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): Answer to the Input Questions about the DataFrame

        """

        self._start_time = time.time()

        self.log(f"Running PandasAI with {self._llm.type} LLM...")

        self._prompt_id = str(uuid.uuid4())
        self.log(f"Prompt ID: {self._prompt_id}")

        try:
            if self._enable_cache and self._cache and self._cache.get(prompt):
                self.log("Using cached response")
                code = self._cache.get(prompt)
            else:
                rows_to_display = 0 if self._enforce_privacy else 5

                multiple: bool = isinstance(data_frame, list)

                # MOD
                if multiple:
                    if df_head:
                        heads = df_head
                    else:
                        heads = [
                            anonymize_dataframe_head(dataframe)
                            if anonymize_df
                            else dataframe.head(rows_to_display)
                            for dataframe in data_frame
                    ]
                    

                    multiple_dataframes_instruction = self._non_default_prompts.get(
                        "multiple_dataframes", MultipleDataframesPrompt
                    )
                    code = self._llm.generate_code(
                        multiple_dataframes_instruction(dataframes=heads),
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": heads,
                    }

                else:
                    # MOD
                    if not df_head:
                        df_head = data_frame.head(rows_to_display)
                        if anonymize_df:
                            df_head = anonymize_dataframe_head(df_head)

                    generate_code_instruction = self._non_default_prompts.get(
                        "generate_python_code", GeneratePythonCodePrompt
                    )(
                        prompt=prompt,
                        df_head=df_head,
                        num_rows=data_frame.shape[0],
                        num_columns=data_frame.shape[1],
                    )
                    code = self._llm.generate_code(
                        generate_code_instruction,
                        prompt,
                    )

                    self._original_instructions = {
                        "question": prompt,
                        "df_head": df_head,
                        "num_rows": data_frame.shape[0],
                        "num_columns": data_frame.shape[1],
                    }

                self.last_code_generated = code
                self.log(
                    f"""
                        Code generated:
                        ```
                        {code}
                        ```
                    """
                )

                if self._enable_cache and self._cache:
                    self._cache.set(prompt, code)

            if show_code and self._in_notebook:
                self.notebook.create_new_cell(code)

            for middleware in self._middlewares:
                code = middleware(code)

            answer = self.run_code(
                code,
                data_frame,
                use_error_correction_framework=use_error_correction_framework,
            )
            self.code_output = answer
            self.log(f"Answer: {answer}")

            if is_conversational_answer is None:
                is_conversational_answer = self._is_conversational_answer
            if is_conversational_answer:
                answer = self.conversational_answer(prompt, answer)
                self.log(f"Conversational answer: {answer}")

            self.log(f"Executed in: {time.time() - self._start_time}s")

            return answer
        except Exception as exception:
            self.last_error = str(exception)
            print(exception)
            return (
                "Unfortunately, I was not able to answer your question, "
                "because of the following error:\n"
                f"\n{exception}\n"
            )
    def cleanup_graph_code(self, code):
        return self._llm.call(GraphCleaupPrompt(),
                                 code, 
                                 suffix="\n\nNew Code:\n")
        
    def _retry_run_code(self, code: str, e: Exception, multiple: bool = False):
        """
        A method to retry the code execution with error correction framework.

        Args:
            code (str): A python code
            e (Exception): An exception
            multiple (bool): A boolean to indicate if the code is for multiple
            dataframes

        Returns (str): A python code
        """

        if multiple:
            if isinstance(e, KeyError):
                error_correcting_instruction = ColumnKeyErrorPrompt(
                    code=code,
                    error_returned=e,
                    question=self._original_instructions["question"],
                    df_head=self._original_instructions["df_head"],
                )
            else:
                error_correcting_instruction = self._non_default_prompts.get(
                    "correct_multiple_dataframes_error",
                    CorrectMultipleDataframesErrorPrompt,
                )(
                    code=code,
                    error_returned=e,
                    question=self._original_instructions["question"],
                    df_head=self._original_instructions["df_head"],
                )

        else:
            error_correcting_instruction = self._non_default_prompts.get(
                "correct_error", CorrectErrorPrompt
            )(
                code=code,
                error_returned=e,
                question=self._original_instructions["question"],
                df_head=self._original_instructions["df_head"],
                num_rows=self._original_instructions["num_rows"],
                num_columns=self._original_instructions["num_columns"],
            )

        return self._llm.generate_code(error_correcting_instruction, "")
    
    def get_raw_response(self, 
                         prompt, 
                         data_frame, 
                         anonymize_df=False):
        
        rows_to_display = 0 if self._enforce_privacy else 5

        multiple: bool = isinstance(data_frame, list)

        if multiple:
            heads = [
                anonymize_dataframe_head(dataframe)
                if anonymize_df
                else dataframe.head(rows_to_display)
                for dataframe in data_frame
            ]

            multiple_dataframes_instruction = self._non_default_prompts.get(
                "multiple_dataframes", MultipleDataframesPrompt
            )
            # response = self._llm.generate_code(
            #     multiple_dataframes_instruction(dataframes=heads),
            #     prompt,
            # )
            response = self._llm.call(multiple_dataframes_instruction(dataframes=heads), 
                                 prompt, 
                                 suffix="\n\nCode:\n")

        else:
            df_head = data_frame.head(rows_to_display)
            if anonymize_df:
                df_head = anonymize_dataframe_head(df_head)

            generate_code_instruction = self._non_default_prompts.get(
                "generate_python_code", GeneratePythonCodePrompt
            )(
                prompt=prompt,
                df_head=df_head,
                num_rows=data_frame.shape[0],
                num_columns=data_frame.shape[1],
            )
            # response = self._llm.generate_code(
            #     generate_code_instruction,
            #     prompt,
            # )
            response = self._llm.call(generate_code_instruction, 
                                 prompt, 
                                 suffix="\n\nCode:\n")
        
        return response
        
    
    def generate_code_summary(self, number_dataframes, prompt, code):
        rows_to_display = 0 if self._enforce_privacy else 5

        try:
            response = self._llm.call(
                        CodeSummaryPrompt(
                            # df_head=df.head(),
                            # num_rows=df.shape[0],
                            # num_columns=df.shape[1],
                            number_dataframes=number_dataframes,
                            rows_to_display=rows_to_display,
                            prompt=prompt,
                            code=code,
                        ),
                        prompt,
                        suffix="\n\nAnswer:\n"
                    )
            return response
        except Exception as e:
            return f"Code summary failed to generate because of error: {e}"
    
    def custom_run(self,
        data_frame: Union[pd.DataFrame, List[pd.DataFrame]],
        prompt: str,
        is_conversational_answer: bool = None,
        show_code: bool = False,
        anonymize_df: bool = True,
        use_error_correction_framework: bool = True,
        df_head: Optional[pd.DataFrame] = None,
    ) -> Union[str, pd.DataFrame]:
        try:
            result = self.run(
                data_frame,
                prompt,
                is_conversational_answer,
                show_code,
                anonymize_df,
                use_error_correction_framework,
                df_head=df_head,
            )
        except ExceededMaxRetriesError:
            result = """
            Oops! It seems like there was an issue executing the generated code. 
            Please review your question or try rephrasing it.
            """
        except KeyError:
            result = """
            Oops! We were not able to answer your question with the data provided. Please review your data
            and ensure that the question can be answered with the information provided.
            """

        return result

    def run_code(
        self,
        code: str,
        data_frame: pd.DataFrame,
        use_error_correction_framework: bool = True,
    ) -> str:
        """
        A method to execute the python code generated by LLMs to answer the question
        about the input dataframe. Run the code in the current context and return the
        result.

        Args:
            code (str): A python code to execute
            data_frame (pd.DataFrame): A full Pandas DataFrame
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): String representation of the result of the code execution.

        """

        multiple: bool = isinstance(data_frame, list)

        # Add save chart code
        if self._save_charts:
            code = add_save_chart(code, self._prompt_id, not self._verbose)

        # Get the code to run removing unsafe imports and df overwrites
        code_to_run = self._clean_code(code)
        self.last_code_executed = code_to_run
        self.log(
            f"""
Code running:
```
{code_to_run}
```"""
        )

        environment: dict = self._get_environment()

        if multiple:
            environment.update(
                {f"df{i}": dataframe for i, dataframe in enumerate(data_frame, start=1)}
            )
        else:
            environment["df"] = data_frame

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()) as output:
            count = 0
            while count < self._max_retries:
                try:
                    # Execute the code
                    output = io.StringIO()

                    # Execute the code block and capture the output
                    with contextlib.redirect_stdout(output):
                        exec(code_to_run, environment)
                    code = code_to_run
                    self.last_error = None
                    break
                except Exception as e:
                    if not use_error_correction_framework:
                        raise e
                    count += 1

                    if count == self._max_retries and isinstance(e, KeyError):
                        raise e


                    code_to_run = self._retry_run_code(code, e, multiple)
            
            if count == self._max_retries:
                raise ExceededMaxRetriesError("Exceeded maximum number of retries")

        captured_output = output.getvalue().strip()
        if code.count("print(") > 1:
            return captured_output

        # Evaluate the last line and return its value or the captured output
        # We do this because we want to return the right value and the right
        # type of the value. For example, if the last line is `df.head()`, we
        # want to return the head of the dataframe, not the captured output.
        lines = code.strip().split("\n")
        last_line = lines[-1].strip()

        match = re.match(r"^print\((.*)\)$", last_line)
        if match:
            last_line = match.group(1)

        try:
            result = eval(last_line, environment)

            # In some cases, the result is a tuple of values. For example, when
            # the last line is `print("Hello", "World")`, the result is a tuple
            # of two strings. In this case, we want to return a string
            if isinstance(result, tuple):
                result = " ".join([str(element) for element in result])

            return result
        except Exception:
            return captured_output

    def get_code_output(
        self,
        code: str,
        data_frame: pd.DataFrame,
        use_error_correction_framework: bool = True,
        has_chart: bool = False,
    ) -> str:
        """
        A method to execute the python code generated by LLMs to answer the question
        about the input dataframe. Run the code in the current context and return the
        result.

        Args:
            code (str): A python code to execute
            data_frame (pd.DataFrame): A full Pandas DataFrame
            use_error_correction_framework (bool): Turn on Error Correction mechanism.
            Default to True

        Returns (str): String representation of the result of the code execution.

        """

        multiple: bool = isinstance(data_frame, list)

        # Add save chart code
        if self._save_charts:
            code = add_save_chart(code, self._prompt_id, not self._verbose)

        # Get the code to run removing unsafe imports and df overwrites
        code_to_run = self._clean_code(code)
        self.last_code_executed = code_to_run
        self.log(
            f"""
Code running:
```
{code_to_run}
```"""
        )

        environment: dict = self._get_environment()

        if multiple:
            environment.update(
                {f"df{i}": dataframe for i, dataframe in enumerate(data_frame, start=1)}
            )
        else:
            environment["df"] = data_frame

        # Redirect standard output to a StringIO buffer
        with redirect_stdout(io.StringIO()):
            count = 0
            while count < self._max_retries:
                try:
                    # Execute the code
                    output = io.StringIO()

                    # Execute the code block and capture the output
                    with contextlib.redirect_stdout(output):

                        exec(code_to_run, environment)
                    code = code_to_run
                    break
                except Exception as e:
                    if not use_error_correction_framework:
                        raise e
                    count += 1

                    if count == self._max_retries and isinstance(e, KeyError):
                        raise e

                    code_to_run = self._retry_run_code(code, e, multiple)
            
            if count == self._max_retries:
                raise ExceededMaxRetriesError("Exceeded maximum number of retries")

        captured_output = output.getvalue().strip()

        lines = code.strip().split("\n")
        last_line = lines[-1].strip()

        match = re.match(r"^print\((.*)\)$", last_line)

        result = ""
        # Evaluate the last line and return its value or the captured output only if it isn't a print statement
        # Print statements are captured by default, we print the last line IFF there is no print statement for the final line
        if not match and not has_chart:
            try:
                res_output = io.StringIO()
                with contextlib.redirect_stdout(res_output):
                    eval(f'print({last_line})', environment)
                
                result = res_output.getvalue().strip()
            except Exception:
                pass
        
        return captured_output, result, environment

        
