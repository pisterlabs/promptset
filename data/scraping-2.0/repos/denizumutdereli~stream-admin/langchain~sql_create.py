import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from third_parties.linkedin import scrape_linkedin_profile

load_dotenv()

tables = """
CREATE TABLE public.order_orders (
	id text NOT NULL,
	client_order_id text NOT NULL,
	user_id int8 NOT NULL,
	side text NOT NULL,
	market text NOT NULL,
	price numeric NOT NULL,
	stop_price numeric NULL,
	quantity numeric NOT NULL,
	quote_asset_quantity numeric NULL,
	executed_quantity numeric NULL,
	cumulative_quote_quantity numeric NULL,
	status text NOT NULL,
	time_in_force text NULL,
	match_engine text NOT NULL,
	meta_data text NULL,
	created_at text NULL,
	updated_at text NULL,
	cancelled_at text NULL,
	dust numeric NULL,
	commission numeric NULL,
	"type" text NULL,
	commission_try numeric NULL,
	commission_usdt numeric NULL,
	"__op" text NULL,
	"__table" text NULL,
	"__source_ts_ms" int8 NULL,
	"__deleted" text NULL,
	CONSTRAINT order_orders_pkey PRIMARY KEY (id)
);
CREATE TABLE public.order_trade_orders (
	maker_order_id text NOT NULL,
	taker_order_id text NOT NULL,
	quantity numeric NOT NULL,
	maker_order_price numeric NOT NULL,
	taker_order_price numeric NOT NULL,
	maker_commission numeric NOT NULL,
	taker_commission numeric NOT NULL,
	created_at text NULL DEFAULT '1970-01-01T00:00:00.000000Z'::text,
	maker_commission_try numeric NULL,
	maker_commission_usdt numeric NULL,
	taker_commission_try numeric NULL,
	taker_commission_usdt numeric NULL,
	"__op" text NULL,
	"__table" text NULL,
	"__source_ts_ms" int8 NULL,
	"__deleted" text NULL,
	CONSTRAINT order_trade_orders_pkey PRIMARY KEY (maker_order_id, taker_order_id)
);

"""

openai_api_key = os.getenv("OPENAI_API_KEY")

print(openai_api_key)

if __name__ == "__main__":
    print("hello langchain!")

    summary_template = """
    given the sentence of sql tables {information} I want you to create:
    1. Posgresql SQL
    2. validate your sql with tables and their fields.
    3. use joins when necessary. if not necessary dont join tables
    4. string values are always uppercase and ints are float64 values
    """

    summary_promt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_promt_template)

    while True:
        prompt_input = input("Enter a sentence of SQL tables (or 'exit' to quit): ")

        if prompt_input.lower() == "exit":
            break

        response = chain.run(information=prompt_input)
        print(response)
