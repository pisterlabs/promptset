import openai
import pandas as pd
from io import StringIO
from faker import Faker
from datetime import datetime, timedelta
import random
import getpass
import os
import time

if not os.environ.get("OPENAI_API_KEY"):
    print("OpenAI API Key not found! Please enter your key:")
    os.environ['OPENAI_API_KEY'] = getpass.getpass("Enter your OpenAI key here:")
    
template = pd.read_csv("template.csv")
tableA = pd.read_csv("table_A.csv")
tableB = pd.read_csv("table_B.csv")

date_formats = [
    '%m-%d-%Y',     # mm-dd-yyyy
    '%d-%m-%Y',     # dd-mm-yyyy
    '%Y-%m-%d',     # yyyy-mm-dd
    '%m/%d/%Y',     # mm/dd/yyyy
    '%d/%m/%Y',     # dd/mm/yyyy
    '%Y/%m/%d',     # yyyy/mm/dd
    '%b %d, %Y',    # Mon dd, yyyy
    '%d %b %Y',     # dd Mon yyyy
    '%B %d, %Y',    # Month dd, yyyy
    '%d %B %Y'      # dd Month yyyy
]

fake = Faker()

def askgpt(message, temperature = 1.0, model = "gpt-4"):
  chat_completion = openai.ChatCompletion.create(model=model, temperature = temperature, messages=[{"role": "user", "content": message}])
  return chat_completion.choices[0].message.content


def get_fake_date(format='%m-%d-%Y'):
  end_date = datetime.now()
  start_date = end_date - timedelta(days=5*365)
  fake_date = fake.date_between(start_date=start_date, end_date=end_date)
  return fake_date.strftime(format)

template['Date'] = [get_fake_date() for _ in range(len(template))]
template['EmployeeName'] = [fake.name() for _ in range(len(template))]
template.head()

format = random.choice(date_formats)
datesA = [get_fake_date(format) for _ in range(len(tableA))]
tableA['Date_of_Policy'] = tableA['Policy_Start'] = datesA
names = [fake.name() for _ in range(len(tableA))]
tableA['FullName'] = tableA['Full_Name'] = names
tableA.head()

format = random.choice(date_formats)
tableB['PolicyDate'] = [get_fake_date(format) for _ in range(len(tableB))]
names = [fake.name() for _ in range(len(tableB))]
firstnames = []
lastnames = []
for name in names:
  f,l = name.split(" ")[-2:]
  firstnames.append(f)
  lastnames.append(l)
tableB['Employee_Name'] = names
tableB['Name'] = lastnames
tableB['PlanType'] = firstnames
tableB.head()

def generate_data(template = template, tables = [tableA, tableB], temperature = 1.0):
  msg = f"""

  In the financial sector, one of the routine tasks is mapping data from various sources in Excel tables. For example, a company may have a target format for employee health insurance tables (Template) and may receive tables from other departments with essentially the same information, but with different column names, different value formats, and duplicate or irrelevant columns.

  I'm trying to generate synthetic data to hopefully learn patterns and train a model.  The Template below is a sample .csv table containing the target format that I want to transform tables into.

  After the Template are example tables that can be transformed into the Template's format.  Please create an additional table in the .csv format that can be transformed into the Template's format. Note: the dates don't all have to be in May 2023.

  Template:

  ---

  {template.to_csv(index=None)}

  ---
  """
  for table in tables:
    add_msg = f"""

  Example Table:

  ---

  {table.to_csv(index=None)}

  ---


  """
    msg += add_msg

  msg += """

  Example Table:

  ---
  """

  return askgpt(msg, temperature = temperature)

if __name__ == "__main__":
    test = generate_data(temperature = 1.0)
    pd.read_csv(StringIO(test.strip()), sep=",", header=0).to_csv("{}.csv".format(time.time()), index=None)