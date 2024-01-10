from functools import reduce
import logging
import re
from collections import namedtuple
from langchain import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

from caesura.phases.base_phase import ExecutionOutput, Phase

logger = logging.getLogger(__name__)
relevant_col_tuple = namedtuple("relevant_column", ["table", "column", "contains", "reasons"])


# no improvements with these examples
# 
# EXAMPLE_DISCOVERY = """
# 3.0.1 The 'text_path' column (datatype str) is not relevant to the user's request, because the user didn't query the path and I am not able to open file paths or URLs.
# 3.0.2 No
# 3.1.1 The 'img_url' column (datatype str) is not relevant to the user's request, because the user didn't query the URL and I am not able to access files behind file paths or URLS.
# 3.1.2 No
# 3.2.1 The 'patient_name' column (datatype str) is not relevant, because the user requested to aggregate over patients. Therefore, the patient name is not necessary for the final plot.
# 3.2.2 No
# 3.3.1 The 'patient_report' column (datatype TEXT) is relevant because I am able to read texts in columns of datatype TEXT and the user-queried diagnosis can be extracted from a patient report.
# 3.3.2 Yes, TODO
# 3.4.1 The 'patient_picture' column (datatype IMAGE) is relevant, because I am able to look at images with datatype IMAGE and looking at patient's pictures allows me to determine their gender, which was queried.
# 3.4.2 Yes
# 3.5.1 The 'patient_scan' column (datatype IMAGE) is not relevant, because neither gender nor diagnosis can be determined by looking at the patients' scan.
# 3.5.2 No

# 3.0.1 The 'picture title' (datatype str) is relevant because the user queried for the pictures with the highest number of persons in it. The pictures are identified using their title.
# 3.0.2 Yes
# 3.1.1 The 'picture' (datatype IMAGE) is relevant, because it contains the actual picture necessary to determine the number of persons depicted in it.
# 3.1.2 Yes
# 3.2.1 The 'author' (datatype str) is not relevant, since the user didn't query for the author.
# 3.2.2 No
# 3.3.1 The 'size (kB)' (datatype int) is not relevant, since the user didn't query for the file size.
# 3.3.2 No
# 3.4.1 The 'creation date' (datatype date) is relevant, because the user likes to aggregate by year, which can be determined from the creation date.
# 3.4.2 Yes
# 3.5.1 The 'file path' (datatype str) column is not relevant, since I am not able to follow file paths. The user didn't query for the file path.
# 3.5.2 No

# 3.0.1 The 'img_path' column (datatype str) is not relevant, because I am not able to follow file paths and the user didn't query for the file path.
# 3.0.2 No
# 3.1.1 The 'image' column (datatype IMAGE) is relevant, because it contains the actual images and the user queried for pictures depicting skateboards. Furthermore, the user-queried 'picture perspective' can also be determined by looking at the image.
# 3.1.2 Yes
# """


class DiscoveryPhase(Phase):

    def create_prompts(self):
        result = {}
        for table_name, table in self.database.tables.items():
            chat = ChatPromptTemplate.from_messages([
                self.system_prompt(table),
                HumanMessagePromptTemplate.from_template("My request is: {query}.\n"
                    "In order to plan the steps to do, You must answer the following questions as precisely as possible.\n"
                    "1. If a plot is requested, what kind of plot is most suitable? What would be the best representation for the requested plot?"
                    "   What exactly is plotted on the X and Y-Axis? Answer N/A if not applicable.\n"
                    "2. If a table is requested, what should be the columns of the table? What should be the rows of the table? Answer N/A if not applicable.\n"
                    "{relevance_questions}"
                )
            ])
            result[table_name] = chat
        return result

    def system_prompt(self, table):
        text_columns = ", ".join([c for c in table.get_columns() if table.get_datatype_for_column(c) == "TEXT"])
        image_columns = ", ".join([c for c in table.get_columns() if table.get_datatype_for_column(c) == "IMAGE"])
        result = "You are Data-GPT, and you can tell users which database tables and columns are relevant for a user request. "
        if image_columns:
            result += f"You can look at images ({image_columns}) to determine which and how many objects are depicted in the image, and also determine things like the style and type of the image. "
        if text_columns:
            result += f"You can read and process texts ({text_columns}) to extract information from the text. "
        result += "You cannot open file paths or URLs.\n"

        # result += "You will be asked to answer a set of questions to determine the relevance of tables and columns. Here are some example answers: \n"
        # result += EXAMPLE_DISCOVERY + "\n"
        # result += "Now you have to answer similar questions for this Data:\n"

        result += self.database.describe()
        result += f"Table '{table.name}' =\n" + self.database.peek_table(table, example_text=True)
        return SystemMessagePromptTemplate.from_template(result)

    def get_prompt_global(self, query, relevant_columns):
        system = "You are Data-GPT, and you can tell users which database tables and columns are relevant for a user request. "
        system += f"You can look at images (Datatype IMAGE) to determine which and how many objects are depicted in the image, and also determine things like the style and type of the image. "
        system += f"You can read and process texts (Datatype TEXT) to extract information from the text. "
        system += "You cannot open file paths or URLs.\n"
        system += self.database.describe()
        system_message =  SystemMessagePromptTemplate.from_template(system)

        cols = ", ".join(f"{c.table}.{c.column} ({c.contains} {c.reasons})" for c in relevant_columns)
        human = f"My request is: {query}. I consider these columns as relevant: {cols}. Answer these questions:\n" \
            "1. Are these columns enough to satisfy the request or are there other relevant columns I missed? If they are sufficient, you can skip the other questions.\n" \
            "2. Which other columns are relevant? Please provide them in this format:\n- table_1.col_1: <What does the column contain?>\n...\n- table_n.col_n: <What does the column contain?>\n"
        human_message = HumanMessage(content=human)
        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def get_hint(self, c, pronoun):
        dtype = self.database.get_column_datatype(c.table, c.column) if isinstance(c, relevant_col_tuple) else c
        if dtype == "IMAGE":
            return f"{pronoun} can easily extract structured information from these images."
        if dtype == "TEXT":
            return f"{pronoun} can easily extract structured information from these texts."
        else:
            return f"{pronoun} can transform these values or extract relevant parts, " \
                    "in case they are not yet in the right format."


    def init_chat(self, query, **kwargs):
        return {k: v.format_prompt(query=query, relevance_questions=self.get_relevance_questions(k)).messages
                for k, v in self.create_prompts().items()}

    def execute(self, query, chat_history, **kwargs):
        relevant_columns = RelevantColumns(self.database, query, self.llm) if "relevant_columns" not in kwargs \
            else kwargs["relevant_columns"]

        for table_name, messages in chat_history.items():
            if table_name == "__global__":
                continue
            prompt = ChatPromptTemplate.from_messages(messages)
            llm_chain = LLMChain(llm=self.llm, prompt=prompt)
            ai_output = llm_chain.predict()
            chat_history[table_name].append(AIMessage(content=ai_output))
            cols = self.parse_relevant_columns(table_name, ai_output, self.get_relevance_questions(table_name))
            relevant_columns.extend(cols)

        if "__global__" not in chat_history:
            chat_history["__global__"] = self.get_prompt_global(query, relevant_columns).format_prompt().messages
        prompt = ChatPromptTemplate.from_messages(chat_history["__global__"])

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        ai_output = llm_chain.predict()
        chat_history["__global__"].append(AIMessage(content=ai_output))

        for table, column, desc in re.findall("- (\w+)\.(\w+): (.*)",ai_output.split("\n2.")[-1]):
            if column.endswith("_id"):
                continue
            contains = desc.split(".")[0].split(", which")[0].strip()
            relevant_columns.append(relevant_col_tuple(table, column, contains + ".", ""))

        logger.info(relevant_columns)

        return ExecutionOutput(
            state_update={"relevant_columns": relevant_columns},
            chat_history=chat_history,
        )

    def handle_observation(self, observation, chat_history, **kwargs):
        msg = observation.get_message("Retry answering the above questions!")
        result = {}
        for table_name, messages in chat_history.items():
            new_prompt = ChatPromptTemplate.from_messages([
                *messages,
                msg
            ])
            result[table_name] = new_prompt.messages
        return result

    def reinit_chat(self, observation, chat_history, **kwargs):
        raise NotImplementedError

    def parse_relevant_columns(self, table_name, result, relevance_questions):
        answers = [x.strip() for x in result.split("\n") if x.startswith("3")]
        col_map = [x.split() for x in relevance_questions.split("\n") if x.startswith("3") and x.endswith("?")]
        col_map = {int(x[0].split(".")[1]): x[6] for x in col_map}
        is_relevant_nums = set()
        contains = dict()
        reasons = dict()
        for a in answers:
            num = a.split(" ")[0]
            if num == "3.":
                continue
            i = int(num.split(".")[1])
            answer = a[len(num) + 1:]

            if num.endswith("3") and "yes" in re.split(r"\W", answer.lower()):
                is_relevant_nums.add(i)
            if num.endswith("2"):
                reasons[i] = answer
            elif num.endswith("1"):
                contains[i] = answer.split(".")[0].split(", which")[0].strip() + "."
        result = {relevant_col_tuple(table_name, col_map[k], v, reasons[k]) for k, v in contains.items()
                  if k in is_relevant_nums and k in col_map}
        return result


    def get_relevance_questions(self, table_name):
        result = []
        i = 1
        table = self.database.tables[table_name]
        for i, col in enumerate(table.get_columns()):
            dtype = table.get_datatype_for_column(col)
            if col.endswith("_id"):
                continue
            r = f"3.{i}.1 What is contained in column {col} (table {table_name}, datatype {dtype}) ?\n" \
                f"3.{i}.2 Why or why not is column {table_name}.{col} relevant for the query ({self.get_hint(dtype, 'Data-GPT')})?.\n" \
                f"3.{i}.3 Final decision whether column {table_name}.{col} is relevant (Yes / No).\n" 
            result.append(r)
            i += 1
        return "\n".join(result)


class RelevantColumns(list):
    def __init__(self, database, query, llm):
        self.database = database
        self.query = query
        self.example_values = dict()
        self.llm = llm
        self.relevant_values_via_index = list()
        self.tool_hints = dict()
        self.default_hints = dict()
        super().__init__()

    def __str__(self, with_tool_hints=False, with_join_hints=False, detail_occurrence_limit=2): 
        if len(self) == 0:
            return ""
        hints = self.tool_hints if with_tool_hints else self.default_hints
        self.set_relevant_values_via_index(self.relevant_values_via_index)
        table_occurrences = reduce(lambda a, b: {k: (a.get(k, 0) + b.get(k, 0)) for k in (set(a) | set(b))},
                                   [{c.table: 1} for c in self])
        result = "These columns (among others) are potentially relevant:\n"
        result += "\n".join(
            (f"- The '{c.column}' column of the '{c.table}' table might be relevant. {c.contains} "
             f"These are some relevant values for the column: {self.example_values[c]}." + hints.get(c, ""))
            for c in self if table_occurrences[c.table] <= detail_occurrence_limit
        ) + "\n"
        short_relevant = [f"{c.table}.{c.column} (example values {self.example_values[c][:2]})"
                          for c in self if table_occurrences[c.table] > detail_occurrence_limit]
        if len(short_relevant):
            result += "- The columns " + ", ".join(short_relevant) + " might also be relevant."
        if with_join_hints:
            result += " " + self.get_join_columns()
        return result

    def get_join_columns(self):
        result = set()
        for col in set(c.table for c in self):
            result |= self.dfs(col, frozenset(c.table for c in self))
        return "\n - These are relevant primary / foreign keys: " + ", ".join(f"{t1}.{t2}"
                                                                         for t1, t2 in sorted(result, key=lambda x: x[1]))
    def dfs(self, table, relevant_tables, path=(), visited=frozenset()):
        result = set()
        for link in self.database.tables[table].links:
            neighbor, neighbor_col = next(iter(
                (t.name, c) for t, c  in ((link.table1, link.column1), (link.table2, link.column2)) if t.name != table))
            if neighbor in visited:
                continue
            table, table_col = next(iter(
                (t.name, c) for t, c  in ((link.table1, link.column1), (link.table2, link.column2)) if t.name == table))
    
            step = ((table, table_col), (neighbor, neighbor_col))
            this_path = path + step
            if neighbor in relevant_tables:
                result |= set(this_path)
            else:
                result |= self.dfs(neighbor, relevant_tables, this_path, visited=frozenset(visited | {table}))
        return result

    def with_tool_hints(self):
        return self.__str__(with_tool_hints=True)
    
    def with_join_hints(self):
        return self.__str__(with_join_hints=True)
    
    def __contains__(self, other: object) -> bool:
        inside = {(c.table, c.column) for c in self}
        return (other.table, other.column) in inside

    def append(self, c):
        if c in self or c.table not in self.database.tables or \
                c.column not in self.database.tables[c.table].data_frame.columns:
            return

        super().append(c)
        if c.table not in self.database.tables or c.column not in self.database.tables[c.table].data_frame.columns:
            return
        self.example_values[c] = self.database.tables[c.table].data_frame[c.column][:30].unique()[:10].tolist()
        if self.database.get_column_datatype(c.table, c.column) == "IMAGE":
            self.example_values[c] = self.example_values[c][:3]
            self.default_hints[c] = f" You should look at images in {c.table}.{c.column} to figure out what they depict."
            self.tool_hints[c] = f" Use Visual Question Answering to look at the images in {c.table}.{c.column} and extract information. Use Image Select to filter rows by what is depicted on the images."
        if self.database.get_column_datatype(c.table, c.column) == "TEXT":
            self.example_values[c] = self.example_values[c][:3]
            self.example_values[c] = ["<TEXT>" for _ in self.example_values[c]]
            self.default_hints[c] = f" You should read the texts in {c.table}.{c.column} to figure out what they contain."
            self.tool_hints[c] = f" Use Text Question Answering to read the texts in {c.table}.{c.column} and extract information from them."
        if self.database.has_relevant_values_index(c.table, c.column):
            self.relevant_values_via_index.append(c)

    def extend(self, cols):
        for c in cols:
            self.append(c)

    def set_relevant_values_via_index(self, cols):
        if len(cols) == 0:
            return
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Extract potentially relevant keywords from queries given some column names."),
            HumanMessage(content="Query: Get the place of birth of every US president.\n"
                         "Columns: politician.name, politician.country, politician.rank, politician.place_of_birth"),
            AIMessage(content="politician.country: US, USA, United States of America\npolitician.rank: president"),
            HumanMessage(content="Query: Plot the doses of acetylsalicylic acid for all drugs against "
                         "fever for each producer.\nColumns: drugs.name, drugs.disease, drugs.active_ingredient, drugs.dose"),
            AIMessage(content="drug.disease: fever\ndrugs.active_ingredient: acetylsalicylic acid, ASA"),
            HumanMessagePromptTemplate.from_template("Query: {query}\ncolumns: {columns}")

        ])
        columns = ", ".join(f"{c.table}.{c.column}" for c in cols)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.predict(query=self.query, columns=columns)
        mapping = {(c.table, c.column): c for c in self}
        for line in result.split("\n"):
            if ":" not in line:
                continue
            column, keywords = tuple(x.strip() for x in line.split(":"))
            table, column = column.split(".")
            if table not in self.database.tables or column not in self.database.tables[table].data_frame.columns:
                continue
            keywords = [x.strip() for x in keywords.split(",")]
            values = self.database.get_relevant_values(table, column, keywords)
            self.example_values[mapping[table, column]] = values
        self.relevant_values_via_index = []
