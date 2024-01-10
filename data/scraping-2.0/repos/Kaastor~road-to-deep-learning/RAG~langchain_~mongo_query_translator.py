import os

import openai
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

openai.api_key = os.environ["OPENAI_API_KEY"]

# 1. Prompt to get subset of data
q_template = (
'''
I have a MongoDB collection with the following fields:
```
- Field Name: _id
  Type: String
  Description: unique identifier of document
    
- Field Name: contributors
  Type: Array
  Description: email addresses of users which can read the file

- Field Name: readers
  Type: Array
  Description: email addresses of users which can read the file

- Field Name: created
  Type: Date
  Description: date when file was created

- Field Name: fileSize
  Type: Int64
  Description: size of the file in bytes

- Field Name: mimeType
  Type: String
  Description: mime type of file

- Field Name: owner
  Type: String
  Description: email address of user which owns the file

- Field Name: updated
  Type: Date
  Description: date when file was changed in any way

- Field Name: title
  Type: String
  Description: title of file
```

﻿Based on that information, I want you to perform a two step process:
1. First create a MongoDB query based on the user instruction. Query must correspond to the provided collection fields, and nothing else. I'm interested only in a query itself.
2. Translate this MongoDB query to JSON format I'm going to explain in a moment.
Below you have an example of such format:
```json
 {
  "name": "GeneratedByLLM",
  "type": "DriveFiles",
  "rules": {
   "condition": "AND",
   "rules": [
    {
     "id": "mimeType",
     "field": "mimeType",
     "type": "string",
     "input": "text",
     "operator": "equal",
     "value": "application/x-gzip"
    }
    {
     "condition": "AND",
     "rules": [
      {
       "id": "commenters",
       "field": "commenters",
       "type": "string",
       "input": "text",
       "operator": "is_empty",
       "value": null
      }
     ]
    },
    {
     "condition": "OR",
     "rules": [
      {
       "id": "title",
       "field": "title",
       "type": "string",
       "input": "text",
       "operator": "equal",
       "value": "asdf"
      }
     ]
    }
   ]
  }
 }
```
What you need to know about the translation:
1. Base Structure:
Every rule set will start with the base structure:
```
{
  "name": "RuleSetName",
  "type": "RuleSetType",
  "rules": {
    // ...nested rules and conditions...
  }
}

```
- name and type are string values always with values "GeneratedByLLM" and "DriveFiles"
- `rules` is the actual set of conditions and nested rules.

2. Rules & Conditions:
- The rules field will always contain a primary condition (like AND or OR) and a list of actual rules.

- condition: Can one of the following:
  - "AND": All rules under this condition must be true.
  - "OR": At least one rule under this condition must be true.
  - "NAND": It's the opposite of AND. If both things are true, it's false. Otherwise, it's true.
  - "NOR": It's the opposite of OR. If both things are false, it's true. Otherwise, it's false.
- rules: An array containing the individual rules or nested rule groups

3. Individual Rule:
An individual rule within the rules array can look something like this:
```
{
  "id": "mongoDBFieldName",
  "field": "mongoDBFieldName",
  "type": "mongoDBFieldType",
  "input": "queryValueForGivenField",
  "operator": "mongodbOperator",
  "value": "queryValueForGivenField"
}
```

The components are:

-id: Name of the field from MongoDB collection, for example 'title'.
-field: the same as 'id'.
-input: Describes the type of input. (e.g., text, date, etc.).
-operator: The operation to perform on the field with the value. (e.g., equal, date_equal, is_empty).
-value: The actual value to check the field against.

4. Nested Rule:
Just like our main structure, a nested rule will also have a condition and rules. You can insert these nested rule groups in place of individual rules in the main rule list.

Requirements:
-Keep in mind the nesting limit. In your example, it's two levels deep.
-Always validate your JSON for any structural or syntax errors.
-Try to simplify the rule conditions when possible. Too much nesting can make the rule harder to read and debug.

Instruction from user: {instruction}"
'''
)
prompt = PromptTemplate.from_template(q_template)

# 2. Get the query
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-instruct")
chain = LLMChain(llm=llm, prompt=prompt)

# instruction = "Return me files of user `susan@generalaudittool.com` created before 2023-07-07"
instruction = "db.collection.find({ contributors: { $all: ['przemek@gat.com', 'mateusz@gat.com'] }, mimeType: 'application/pdf' })"

with get_openai_callback() as cb:
    response_query = chain.run(instruction=instruction)
    print("Query: ", response_query)
