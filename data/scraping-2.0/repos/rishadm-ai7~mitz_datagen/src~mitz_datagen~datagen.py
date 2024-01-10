import openai
from faker import Faker

api_key = input("Please enter your OpenAI API key: ")

openai.api_key = api_key

myprompt = input("Please specify your requirement: ")
prompt = myprompt


response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=1024,
  n=1,
  stop=None,
  temperature=0.5,
)

generated_text = response.choices[0].text.strip()

fake = Faker()
table_names = generated_text.split("\n")
fact_table_name = None
dimension_table_names = []
for table_name in table_names:
    column_names = table_name.split(" with ")[0].split(", ")
    num_rows = int(table_name.split(" with ")[1].split(" rows and ")[0])
    num_cols = int(table_name.split(" with ")[1].split(" columns and ")[0])
    table_data = []
    for i in range(num_rows):
        row_data = [fake.format(f"{{0.{j}}}").replace(".", "") for j in range(num_cols)]
        table_data.append(row_data)
    if fact_table_name is None:
        fact_table_name = table_name
        fact_table_data = table_data
        fact_table_columns = column_names
    else:
        dimension_table_names.append(table_name)
        dimension_table_data = table_data
        dimension_table_columns = column_names
        #foreign key column in the fact table that references this dimension table
        foreign_key_column = f"{table_name.split()[0]} ID"
        fact_table_columns.append(foreign_key_column)
        for i in range(num_rows):
            fact_table_data[i].append(str(i+1))
        #primary key column in the dimension table
        dimension_table_columns.insert(0, "ID")
        for i in range(num_rows):
            dimension_table_data[i].insert(0, str(i+1))

#print fact table
print(fact_table_name + ":")
print(", ".join(fact_table_columns))
for row_data in fact_table_data:
    print(", ".join(row_data))

#print dimension tables
for i in range(len(dimension_table_names)):
    print(dimension_table_names[i] + ":")
    print(", ".join(dimension_table_columns))
    for row_data in dimension_table_data:
        print(f"{i+1}, " + ", ".join(row_data[1:]))
