import openai
import secrets_config
import config
import json
import heapq

openai.api_key = secrets_config.openapikey


def autocomplete_from_docs() -> list:
    return [
        "SELECT 'Hello World';\n",
        "SHOW FUNCTIONS;\n",
        "SELECT CURRENT_TIMESTAMP;\n",
        "CREATE TABLE employee_information (\n    emp_id INT,\n    name VARCHAR,\n    dept_id INT\n) WITH ( \n    'connector' = 'filesystem',\n    'path' = '/path/to/something.csv',\n    'format' = 'csv'\n);\n",
        "SELECT * from employee_information WHERE dept_id = 1;\n",
        "SELECT \n   dept_id,\n   COUNT(*) as emp_count \nFROM employee_information \nGROUP BY dept_id;\n",
        "INSERT INTO department_counts\nSELECT \n   dept_id,\n   COUNT(*) as emp_count \nFROM employee_information;\n"
    ]


def autocomplete_with_gpt() -> list:
    return [
        "SELECT * FROM Orders\nINNER JOIN Product\nON Orders.productId = Product.id\n",
        "SELECT *\nFROM Orders\nINNER JOIN Product\nON Orders.product_id = Product.id\n",
        "SELECT *\nFROM Orders\nLEFT JOIN Product\nON Orders.product_id = Product.id\n\nSELECT *\nFROM Orders\nRIGHT JOIN Product\nON Orders.product_id = Product.id\n\nSELECT *\nFROM Orders\nFULL OUTER JOIN Product\nON Orders.product_id = Product.id\n",
        "SELECT *\nFROM Orders o, Shipments s\nWHERE o.id = s.order_id\nAND o.order_time BETWEEN s.ship_time - INTERVAL '4' HOUR AND s.ship_time\n",
    ]


def explanation_with_gpt() -> str:
    return "This is a mocked explanation from ChatGPT that explains your query!"
