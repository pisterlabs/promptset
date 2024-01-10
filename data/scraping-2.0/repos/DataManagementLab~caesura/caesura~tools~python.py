import re
import importlib
import logging
from langchain import LLMChain, PromptTemplate
from caesura.database.database import Database, Table
from caesura.observations import Observation
from caesura.tools.base_tool import BaseTool
from caesura.observations import ExecutionError
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate


logger = logging.getLogger(__name__)

IMPORTS = ["pandas", "datetime", "numpy", "re"]
IMPORT_REGEX = r"(from \w+ |)import (\w+)( as \w+|)"

class TransformTool(BaseTool):
    name = "Python"
    description = (
        "Transform the values for a certain column using python. Can be used to transform dates, numbers etc, or to extract data from semi-structured documents. "
        "Three input arguments: (column to transform; new name for the transformed column; natural language explanation of the python code to be executed). "
        "For example: (date_of_birth; month_of_birth; extract the month from the dates) "
        "Cannot deal columns of type IMAGE or TEXT. Cannot filter rows. Has only access to the libraries " + ",".join(IMPORTS) + ".\n"
    )
    args = ("column to transform", "new name for the transformed column", "natural language explanation of the python code to be executed")

    def __init__(self, database: Database, llm, interactive: bool):
        super().__init__(database)
        self.llm = llm
        self.interactive = interactive

    def run(self, tables, input_args, output):
        """Use the tool."""
        table = tables[0]
        column, new_name, explanation = tuple(input_args)
        if "." in column:
            table, column = column.split(".")
        
        ds = self.database.get_table_by_name(table)
        if column in ds.image_columns:
            raise ExecutionError(description="Python cannot be called on columns of IMAGE datatype. "
                                 "For these columns, use the other tools, e.g. Visual Question Answering")
        if column in ds.text_columns:
            raise ExecutionError(description="Python cannot be called on columns of TEXT datatype. "
                                 "For these columns, use the other tools, e.g. Text Question Answering.")
        df, func_str = self.execute_python(ds, column, new_name, explanation)
        result = Table(
            output if output is not None else table, df,
            f"Result of Python: table={table}, column={column}, new_column={new_name}, code={explanation}",
            parent=ds
        )

        # Add the result to the working memory
        observation = self.database.register_working_memory(result, peek=[new_name])
        observation = Observation(description=observation, plan_step_info=func_str)
        return observation

    def execute_python(self, ds, column, new_name, explanation):
        if column not in ds.data_frame.columns:
            raise ExecutionError(description=f"Column {column} does not exist in table {ds.name}.")
        chat_thread = []
        i = 0
        while True:
            try:
                func, dtype, func_str = self.get_func(explanation, ds.data_frame[column][:10],
                                                      chat_thread=chat_thread, column=column, new_column=new_name)
                df = ds.data_frame.copy()
                df[new_name] = df[column].apply(func).astype(dtype)
                return df, func_str
            except Exception as e:
                if i >= 3:
                    raise ExecutionError(description="Python tool failed. Use another tool!")
                chat_thread = self.handle_errors(chat_thread=chat_thread, error=e, request=explanation)
                i += 1

    def get_func(self, explanation, data, chat_thread, column, new_column):
        modules = ", ".join(IMPORTS)
        params = dict(explanation=explanation, data=str(data), modules=modules, column=column, new_column=new_column)

        if len(chat_thread) == 0:  # Start of conversation
            chat_thread.append(
                HumanMessagePromptTemplate.from_template(
                    "{explanation}:\n```py\n>>> print({column}[:10])\n{data}\n```\n"
                    "It is a pandas Series object. Please call the 'apply' method with a lambda expression, "
                    "and make sure to always call astype() in the same line. Assign the result to a variable called '{new_column}'. "
                    "Template to use: `{new_column} = {column}.apply(lambda x: <code>).astype(<dtype>)`. You can use {modules}."
            ).format(**params))

        prompt = ChatPromptTemplate.from_messages(chat_thread)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.predict().strip()
        chat_thread.append(AIMessage(content=result))
        match = re.search(fr"{new_column} = (\w+\[\"|\w+\['|){column}(\"\]|'\]|)\.apply\((.*)\)\.astype\((.*)\)", result)
        if match is None:
            raise ValueError(f"Use correct template: `{new_column} = {column}.apply(lambda x: <code>).astype(<dtype>)`")

        code, dtype = match[3], match[4]
        functions = self.parse_function_definitions(result)

        function_str = "\n".join(functions)
        function_str = f"{function_str}{column}.apply({code}).astype({dtype})"
        if self.interactive and not next(iter(
            input(f"\nSecurity-Check: Is >>> {function_str} <<< fine (Y,n) ? > ")
        ), "y").lower() == "y":
            exit(0)
        loc = self.manage_imports(result, functions)

        func = eval(code, loc)  # get function handler
        dtype = eval(dtype, loc)
        return func, dtype, function_str

    def parse_function_definitions(self, result):
        functions = list()
        for m in re.finditer(r"( *)def (\w+)\(.*\):.*(\n\1    .*)+", result):
            indent = len(m[1])
            func = "\n".join(l[indent:] for l in m[0].split("\n"))
            if not re.search(IMPORT_REGEX, func):
                functions.append(func + "\n")
        return functions

    def manage_imports(self, result, functions):
        if "```" in result:
            result = result.split("```")[1]
        loc = {m: importlib.import_module(m) for m in IMPORTS}
        for from_stmt, module, alias in re.findall(IMPORT_REGEX, result):
            from_stmt = [x for x in from_stmt[5:].strip().split(".") if x]
            alias = alias[4:].strip() or module
            module = from_stmt + [module]

            target = loc[module[0]]
            for m in module[1:]:
                target = getattr(target, m)

            loc[alias] = target
            for f in functions:
                exec(f, loc)
        return loc

    def handle_errors(self, chat_thread, error, request):
        error_str = f"{type(error).__name__}({error})"
        code = re.search(fr"\w+ = (\w+\[\"|\w+\['|)\w+(\"\]|'\]|)\.apply\((.*)\)\.astype\((.*)\)", chat_thread[-1].content)
        code = code[0] if code is not None else "<could not parse code with template>"
        prompt = ChatPromptTemplate.from_messages([
            *chat_thread,
            HumanMessagePromptTemplate.from_template(
                "Something went wrong executing `{code}`. This is the error I got: {error}. "
                "Can you answer me these four questions:\n"
                "1. What is the reason for the error?\n"
                "2. Is there another way to '{request}', potentially using another library (from {libraries})?\n."
                "3. Can this be fixed? Or is there something wrong in my request? Answer 'Yes' if it can be fixed, and 'No' otherwise.\n"
                "4. If it can be fixed, how can it be fixed? If it cannot be fixed, please explain the error and why it cannot be implemented using python.\n"
                "5. Write fixed code, if possible."
            ),
            AIMessage(content="I'm sorry that the executed code failed. Here are the answers to the questions:\n1.")
        ])
        libraries = ", ".join(IMPORTS)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        logger.warning(prompt.format(error=error_str, code=code, libraries=libraries, request=request))
        result = chain.predict(error=error_str, code=code, libraries=libraries, request=request, stop=["\n5."])
        logger.warning(result)
        explanation, _, can_be_fixed, description  = \
            tuple(x.strip() for x in re.split(r"(\n2\.|\n3\.|\n4\.)", result)[slice(0, None, 2)])
        
        can_be_fixed = "yes" in re.split(r"\W", can_be_fixed.lower())
        if "```" in description:
            description = description.split("```")[0]
            description = ".".join(description.split(".")[:-1]) + "."
        if not can_be_fixed:
            raise ExecutionError(description=description, original_error=error)

        prompt = ChatPromptTemplate.from_messages([
            *chat_thread,
            HumanMessagePromptTemplate.from_template(
                "Something went wrong executing `{code}`. This is the error I got: {error}. "
                "{explanation} Please fix it, but make sure you adhere to the template! This is how you could do it: {fix_idea}"
            ),
        ]).format_prompt(error=error_str, code=code, explanation=explanation, fix_idea=description)
        return prompt.messages
