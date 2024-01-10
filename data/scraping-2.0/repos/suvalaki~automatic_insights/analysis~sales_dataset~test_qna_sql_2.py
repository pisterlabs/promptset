from typing import List, Any

from pydantic import BaseModel

from ai.chains.sql.objective_evaluation.name_evaluation.database_filter import (
    TableSelectionChain,
)
from ai.chains.sql.objective_evaluation.name_evaluation.bfs_filter import (
    SingleTablenameRelevanceEvaluationChain,
)
from ai.chains.sql.objective_evaluation.name_evaluation.bfs_filter import (
    MultipleTablenameRelevanceEvaluationChain,
)

from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI


model_name = "gpt-3.5-turbo"
model = ChatOpenAI(model_name=model_name, temperature=0.0)


db = SQLDatabase.from_uri("sqlite:///./analysis/sales_dataset/data/data.db")

chain = TableSelectionChain(llm=model, db=db)

# reply = chain.predict(objective="What are the number of customers in the UK.")

# print(reply.json(indent=2))


from typing import Dict, Any
from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler


prompt = "Critique the output: "


class CritiqueLLMCallback(BaseCallbackHandler, LLMChain):
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        pass


single_chain = SingleTablenameRelevanceEvaluationChain(llm=model)
single_chain.predict(
    objective="What are the number of customers in the UK.",
    table="retail",
    table_info=db.get_table_info_no_throw(["retail"]),
    tables=["calendar", "retail", "population"],
)

single_chain.predict(
    objective="What are the number of customers in the UK.",
    table="calendar",
    table_info="",
    tables=["calendar", "retail", "population"],
)


# Rate which tables we think are relevant for answering the objective
multiple_chain = MultipleTablenameRelevanceEvaluationChain(llm=model, db=db)
evaluations = multiple_chain.predict(
    objective="What are the number of customers in the UK.",
    tables=["calendar", "retail", "population"],
)


# Setup a critique for the outputs?


# Now filter to tables with high enough score. Provide those
# tables to the query planner with their columns.
# Do we augment each column with a potential description?


# def dummy_filter(obj):
#     return [o for o in obj if o.score > 0.7]


# filtered = dummy_filter(evaluations)


from ai.chains.sql.schema_evaluation.column.column_single import (
    SingleColumnEvaluateDescriptionChain,
    EnhanceInputs,
    ColumnInfoInputsListExtractor,
    MultipleColumnDescriptionChain,
)

colum_describer = SingleColumnEvaluateDescriptionChain(llm=model)
multiple_column_describer = MultipleColumnDescriptionChain(
    llm=model,
    db=db,
    # augment_inputs=EnhanceInputs(db),
    # extract_inputs=ColumnInfoInputsListExtractor(db),
)

reply = multiple_column_describer.predict(
    table_name="retail",
)


from ai.chains.sql.schema_evaluation.table.table_llm import (
    TableEvaluateDescriptionChain,
    TableEvaluationChain,
)

# For a given table get the information
table_describer = TableEvaluateDescriptionChain(llm=model)
describer = TableEvaluationChain(
    column_description_chain=multiple_column_describer,
    table_description_chain=table_describer,
    return_column_descriptions=True,
)

reply = describer.predict(
    table_name="retail",
)

# Given the objective and relevant tables (explored)
# evaluate if a query can solve the problem.
# OR create a plan for building multiple queries
# to find an answer ... o

# 1. Given this objective and these tables does it look
#    like an answer is possible using the data
# 2. Given this objective plan out one or more select
#    queries that answer the objective.
# Returned value should just be the query.
# We should really only ever need a single query tbh
# maybe make it use WITH statements alongside COT style
# reasoning to plan the query.

from ai.chains.sql.objective_evaluation.table_llm_objective import (
    create_objective_evaluate_table_chain,
)

table_explainer_chain = create_objective_evaluate_table_chain(llm=model)

table_eval = table_explainer_chain.predict(
    objective="What are the number of customers in the UK.",
    table_name="retail",
    infered_columns=reply.json(),
)

from ai.chains.sql.objective_evaluation.column_llm_objective import (
    create_objective_evaluate_column_chain,
)

column_explainer_chain = create_objective_evaluate_column_chain(llm=model)

# InvoiceId
tbl = "retail"
col = reply.columns[0].column
inputs = describer.column_description_chain.input_prepper({"table_name": tbl})
additional = describer.column_description_chain.extract_inputs(inputs)["extracted"][0]
column_eval = column_explainer_chain.predict(
    objective="What are the number of customers in the UK.",
    table_name="retail",
    column_name=col,
    table_info=inputs["table_info"],
    column_extract=additional["column_extract"],
    infered_columns="",  # reply.json()
)

# Quanitity
tbl = "retail"
col = reply.columns[3].column
inputs = describer.column_description_chain.input_prepper({"table_name": tbl})
additional = describer.column_description_chain.extract_inputs(inputs)["extracted"][3]
column_eval = column_explainer_chain.predict(
    objective="What are the number of customers in the UK.",
    table_name="retail",
    column_name=col,
    table_info=inputs["table_info"],
    column_extract=additional["column_extract"],
    infered_columns="",  # reply.json()
)

# CustomerId (YES)
tbl = "retail"
col = reply.columns[6].column
inputs = describer.column_description_chain.input_prepper({"table_name": tbl})
additional = describer.column_description_chain.extract_inputs(inputs)["extracted"][6]
column_eval = column_explainer_chain.predict(
    objective="What are the number of customers in the UK.",
    table_name="retail",
    column_name=col,
    table_info=inputs["table_info"],
    column_extract=additional["column_extract"],
    infered_columns="",  # reply.json()
)


from ai.chains.sql.objective_evaluation.objective_llm import (
    create_sql_db_evaluator_chain,
)

sql_eval_chain = create_sql_db_evaluator_chain(
    db=db,
    llm=model,
    # name_evaluator=multiple_chain,
    # table_evaluator=table_explainer_chain,
)

z = sql_eval_chain(dict(objective="How many customers are in the UK?"))


objective_name_evals = sql_eval_chain("How many customers are there in the UK?")
tbls = [t.table for t in objective_name_evals["tablename_evaluations"]]


from ai.chains.sql.schema_evaluation.table.multiple_table_llm import (
    MultipleTableEvaluatorChain,
)

multiple_table_eval = MultipleTableEvaluatorChain(chain=describer)

multiple_table_eval.predict(objective="an objective", tables=tbls)
