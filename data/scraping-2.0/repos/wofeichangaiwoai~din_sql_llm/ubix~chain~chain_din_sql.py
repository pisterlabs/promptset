
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pdb
from langchain.prompts import PromptTemplate
from pydantic import Extra
from sqlalchemy.engine import create_engine
from trino.sqlalchemy import URL
from sqlalchemy import *
from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

from ubix.common.llm import get_llm

easy_template = """
# Use the the schema links to generate the SQL queries for each of the questions.
Q: "Find the buildings which have rooms with capacity more than 50."
Schema_links: [classroom.building,classroom.capacity,50]
SQL: SELECT DISTINCT building FROM classroom WHERE capacity  >  50

Q: "Find the room number of the rooms which can sit 50 to 100 students and their buildings."
Schema_links: [classroom.building,classroom.room_number,classroom.capacity,50,100]
SQL: SELECT building ,  room_number FROM classroom WHERE capacity BETWEEN 50 AND 100

Q: "Give the name of the student in the History department with the most credits."
Schema_links: [student.name,student.dept_name,student.tot_cred,History]
SQL: SELECT name FROM student WHERE dept_name  =  'History' ORDER BY tot_cred DESC LIMIT 1

Q: "Find the total budgets of the Marketing or Finance department."
Schema_links: [department.budget,department.dept_name,Marketing,Finance]
SQL: SELECT sum(budget) FROM department WHERE dept_name  =  'Marketing' OR dept_name  =  'Finance'

Q: "Find the department name of the instructor whose name contains 'Soisalon'."
Schema_links: [instructor.dept_name,instructor.name,Soisalon]
SQL: SELECT dept_name FROM instructor WHERE name LIKE '%Soisalon%'

Q: "What is the name of the department with the most credits?"
Schema_links: [course.dept_name,course.credits]
SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY sum(credits) DESC LIMIT 1

Q: "How many instructors teach a course in last year?"
Schema_links: [teaches.ID,teaches.semester,teaches.YEAR,Spring,2022]
SQL: SELECT COUNT (DISTINCT ID) FROM teaches WHERE semester  =  'Spring' AND YEAR  =  2022

Q: "Find the name of the students and their department names sorted by their total credits in ascending order."
Schema_links: [student.name,student.dept_name,student.tot_cred]
SQL: SELECT name ,  dept_name FROM student ORDER BY tot_cred

Q: "Find the year which offers the largest number of courses."
Schema_links: [SECTION.YEAR,SECTION.*]
SQL: SELECT YEAR FROM SECTION GROUP BY YEAR ORDER BY count(*) DESC LIMIT 1

Q: "What are the names and average salaries for departments with average salary higher than 42000?"
Schema_links: [instructor.dept_name,instructor.salary,42000]
SQL: SELECT dept_name ,  AVG (salary) FROM instructor GROUP BY dept_name HAVING AVG (salary)  >  42000

Q: "How many rooms in each building have a capacity of over 50?"
Schema_links: [classroom.*,classroom.building,classroom.capacity,50]
SQL: SELECT count(*) ,  building FROM classroom WHERE capacity  >  50 GROUP BY building

Q: "Find the names of the top 3 departments that provide the largest amount of courses?"
Schema_links: [course.dept_name,course.*]
SQL: SELECT dept_name FROM course GROUP BY dept_name ORDER BY count(*) DESC LIMIT 3

Q: "Find the maximum and average capacity among rooms in each building."
Schema_links: [classroom.building,classroom.capacity]
SQL: SELECT max(capacity) ,  avg(capacity) ,  building FROM classroom GROUP BY building

Q: "Find the title of the course that is offered by more than one department."
Schema_links: [course.title]
SQL: SELECT title FROM course GROUP BY title HAVING count(*)  >  1

Q: "What is the maximum avenue in last three year?"
Schema_links: [product.avenue, 2022, 2019, product.date]
SQL: SELECT max(avenue) from product where date between '2019-01-01 00:00:00' and '2022-12-31 23:59:59'

Q: "What is the minimum price in last year?"
Schema_links: [food.price, food.year, 2022]
SQL: SELECT min(price) from food where year = 2022 ?

Q: "What are the names for classrooms with average students higher than 20?"
Schema_links: [classrooms.names,classrooms.students,20]
SQL: SELECT names FROM classrooms  HAVING AVG (students)  >  20

Q: "What are our revenues for the past 3 years?":
Schema_links: [invoice_header.totalnetamount,invoice_header.creationdatetime]
SQL: SELECT sum(totalnetamount) FROM invoice_header where creationdatetime between '2019-01-01 00:00:00' and '2022-12-31 23:59:59'

Table invoice_items, columns = [parentobjectid,netamount,description,productid,partyuuid]

Table invoice_header, columns = [objectid,creationdatetime,totalnetamount]

Table customercommon, columns = [parentobjectid,businesspartnername]

Table customer, columns = [objectid,uuid,]

Q: "{input}"
schema_links: [{schema_links}]
SQL:
"""

hard_template = """
Q: "Find the title of courses that have two prerequisites?"
schema_links: [course.title,course.course_id = prereq.course_id]
SQL: SELECT T1.title FROM course AS T1 JOIN prereq AS T2 ON T1.course_id  =  T2.course_id GROUP BY T2.course_id HAVING count(*)  =  2

Q: "What are salaries by employees for the past three years?"
schema_links: [company.salary,company.employee_id=employee.employee_id,company.YEAR]
SQL: SELECT sum(T1.salary) FROM company AS T1 JOIN employee AS T2 ON T1.employee_id = T2.employee_id WHERE T1.YEAR  between '2019-01-01 00:00:00' and '2022-12-31 23:59:59'

Q: "Find the name of students who took any class in the years of 2009"
Schema_links: [student.name,student.id = takes.id,takes.YEAR,2009]
SQL: SELECT DISTINCT T1.name FROM student AS T1 JOIN takes AS T2 ON T1.id  =  T2.id WHERE T2.YEAR  =  2009

Q: "Find the total number of students and total number of instructors for each department."
Schema_links: [department.dept_name = student.dept_name,student.id,department.dept_name = instructor.dept_name,instructor.id]
SQL: SELECT count(DISTINCT T2.id) ,  count(DISTINCT T3.id) ,  T3.dept_name FROM department AS T1 JOIN student AS T2 ON T1.dept_name  =  T2.dept_name JOIN instructor AS T3 ON T1.dept_name  =  T3.dept_name GROUP BY T3.dept_name

Table invoice_items, columns = [parentobjectid,netamount,description,productid,partyuuid]

Table invoice_header, columns = [objectid,creationdatetime,totalnetamount]

Table customercommon, columns = [parentobjectid,businesspartnername]

Table customer, columns = [objectid,uuid,]

Q: "{input}"
Schema_links: [{schema_links}]
SQL:
"""
join_template = [["invoice_items.parentobjectid=invoice_header.objectid"], \
                 ["customer.uuid=invoice_header.partyuuid", "customercommon.parentobjectid=customer.objectid"]]

def transform(query):
    query = query.replace("our ", "")
    query = query.replace("What are", "sum up")
    if "by product" in query:
        query = query.replace("revenues by product", "netamount by productid")
        query += " creationdatetime"
    elif "by customer" in query:
        query = query.replace("revenues", "totalnetamount by businesspartnername")
        query += " creationdatetime"
    else:
        query = query.replace("revenues", "totalnetamount")
    if query == "Can you show me monthly income statements for the last 12 months":
        query = "sum up totalnetamount for the past year"
    return query

def get_schema_links(query):
    query = transform(query)
    is_hard = False
    template_s = easy_template.split("\n")
    query_l = query.replace("?", "").split(" ")
    dct = {}
    for item in template_s:
        if "Table" in item:
            item_s = item.split(",")
            key = item_s[0].replace("Table ", "")
            val_l = []
            for val in item_s[1:]:
                val_strip = val.replace(" columns = [", "").replace("]", "")
                if val_strip in query_l:
                    val_l.append(val_strip)
            if len(val_l) > 0:
                dct[key] = val_l

    result = []
    key_set = set()

    for key in dct:
        key_set.add(key)
        for val in dct[key]:
            result.append(key + "." + val)
    if len(key_set) > 1:
        if "businesspartnername" not in query:
            for temp in join_template[0]:
                result.append(temp)
        else:
            for temp in join_template[1]:
                result.append(temp)
        is_hard = True
    return str(result).replace(" ", ""), is_hard

PROMPT = PromptTemplate(
    input_variables=["input", "schema_links"],
    template=easy_template,
)

def format(sql):
    if "date" in sql:
        result = sql.replace("\'20", "TIMESTAMP \'20").replace("\"20", "TIMESTAMP \"20")
        return result
    else:
        return sql

def hive_search(sql):
    hive_host = "trino"
    port = 8080
    user_name = "hive"
    catalog="hive"
    #hive_database = "65057500bed4c2ac869fe964"
    #hive_database = "654a464f1dd821018bd47cd5"
    hive_database = "65487027ce6b72312eff28a2"

    engine = create_engine(
        URL(
            host=hive_host,
            port=port,
            user=user_name,
            catalog=catalog,                                                                                                                                                                             schema=hive_database,
        ),
    )
    with engine.connect() as con:
        sql = format(sql)
        #sql = "select count(*) from Invoice_Header;"
        print(sql)
        result_t = con.execute(text(sql))
        result = result_t.fetchall()
    return result


class DIN_Chain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables[:1]

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        schema_links, is_hard = get_schema_links(inputs.get("input"))
        #schema_links =  "[invoice_header.totalnetamount,invoice_header.creationdatetime]"
        inputs["schema_links"] = schema_links
        if is_hard:
            self.prompt.template = hard_template
        else:
            self.prompt.template = easy_template
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )
        #print(f'Original response:{response}')
        text = response.generations[0][0].text
        #pdb.set_trace()
        #print(f'Final Text:{text}')
        result_ = text.split("\n\n")[0]
        result_ = result_.replace("\n", " ")
        print(f'🔴Final SQL:{result_}\n=====')
        final = hive_search(result_)

        res = []
        for item in final:
            dct_item = {}
            try:
                dct_item["memory_type"] = str(item[0])
                dct_item["memory"] = str(item[1])
            except:
                pass
            res.append(dct_item)
        dct_final = res
        dct = {}
        dct["sql"] = result_
        dct["answer"] = dct_final
        print("dct", dct)
        print("============================")
        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key:dct}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        #print(f'DummyChain:{inputs}')
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"


def get_din_chain(llm):
    chain = DIN_Chain(llm=llm, prompt=PROMPT)
    return chain


if __name__ == "__main__":
    chain = get_din_chain(get_llm())
    #query = "What are our revenues for the past 3 years"
    #query = "What are our revenues by product for the past year"
    #query = "What are our revenues by customer for the past 3 years"
    query = "Can you show me our monthly income statements for the last 12 months"
    print("query:", query)
    result = chain.run(query)
    print(result)

"""
PYTHONPATH=. LLM_TYPE=din python ubix/chain/chain_din_sql.py
"""
