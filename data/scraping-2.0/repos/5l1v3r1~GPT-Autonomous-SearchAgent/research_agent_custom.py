import openai
import arxiv

# Set up the OpenAI API
openai.api_key = "sk-" # Replace the string content with your OpenAI API key

"""
Wrap the OpenAI API call in this function
"""

def getResponse(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,  # We want consistent behavior, so we set a very low temperature
        messages=[
            {"role": "system", "content": "You're a helpful assistant. Carefully follow the user's instructions."},
            {"role": "user", "content": prompt}
        ]
    )
    response = str(response['choices'][0]['message']['content'])
    return response


"""
Use GPT to determine the action to take by giving it the objective, memory, and tools.
If it think it has finished the objective, just give the answer.
If it needs more info, it will pick the tool to get the relevant information based on the tool description.
"""


def determineAction(objective, memory, tools):
    formattedPrompt = f"""Determine if the following memory is enough to answer\n
    the user's objective. Your past actions are stored in the memory for reference\n
    If it is enough, answer the question in the format: 'FINAL ANSWER: '. \n
    If the memory is not enough, you can use a tool in the available tools section\n
    to get more information. When using a tool you should use this format: \n
    'USE :'. If no tool can help you achieve the user's \n
    objective, then answer 'FINAL: CANNOT ANSWER'.

    ```Objective
    Answer: {objective}
    ```

    ```Memory
    {memory}
    ```

    ```Available Tools
    {tools}
    ```

    """
    response = getResponse(formattedPrompt)
    (finished, result, memory) = parseResponse(response, memory, tools)
    return (finished, result, memory)


"""
Parse the response from GPT to determine if the objective is finished.
If it is finished, just give the final answer.
If the objective cannot be finished with the context and tools, it will say it cannot answer
If GPT picks a tool, execute the tool and save the result of the tool in memory.
"""


def parseResponse(response, memory, tools):
    finished = False

    if response.startswith('FINAL ANSWER:'):
        finished = True
        memory.append(response)
        return (finished, response, memory)
    elif response == 'FINAL: CANNOT ANSWER':
        finished = True
        memory.append(response)
        return (finished, response, memory)
    elif response.startswith('USE:'):
        # split the string using ':' as the delimiter
        parsed_str = response.split(':')

        # 'USE: searchArxiv with the search key word "ReAct reasoning and acting in language models" to gather more information.'
        # get the tool name and parameter
        tool_name = parsed_str[1].split(" ")[1]
        parameter = parsed_str[1]

        print("THOUGHT: " + response)
        memory.append("THOUGHT: " + response)

        result = executeTool(tool_name, parameter, tools)

        new_memory = "OBSERVATION: " + str(result)
        print(new_memory)
        memory.append(new_memory)

        return (finished, result, memory)


"""
Execute the tool that GPT picks using the parameter it gives.
Returns the execution result so that GPT can have the relevant info.
"""


def executeTool(tool_name, parameter, tools):
    # Find the tool with the given name
    tool = None
    for t in tools:
        if t['tool_name'] == tool_name:
            tool = t
            break

    # If the tool is found, execute its function with the given parameter
    if tool:
        return tool['function_name'](parameter)
    else:
        return "Tool not found"


"""
Wrap the search arxiv function as a tool for GPT
Input is a search keyword
Output is a list of dictionaries with title, published date, authors, and summary of papers
"""


def searchArxiv(keyword):
    # Perform a search with the given query
    search = arxiv.Search(query=keyword, max_results=3)

    # Get the metadata for each result and extract relevant information
    results = []
    for result in search.results():
        title = result.title
        published_date = result.published.strftime("%Y-%m-%d")
        authors = ", ".join(author.name for author in result.authors)
        summary = result.summary

        # Store the extracted information as a dictionary
        results.append((
            "title: " + title,
            "published_date: " + published_date,
            "authors: " + authors,
            "summary: " + summary
        ))

    # Return the list of tuples containing the result information
    return results


"""
Initialize memory, tools for the GPT agent.
Ask for a user objective and let it run iteratively untill the objective is achieved.
As a safety measure, it will also stop after 5 iterations just in case things go wrong.
"""


def startAgent():
    objective = input("What is your research question? ")
    # For simplicity, we will just use a list to store every thing.
    # For production, you will probably use vector databases.
    memory = []

    tools = [{'tool_name': 'searchArxiv',
              'description': """You can use this tool to search for scientific papers on Arxiv. The response will have title, author, published date, and summary.""",
              'function_name': searchArxiv,
              'parameter': 'search key word'}]

    n = 0
    while True:
        (finished, result, memory) = determineAction(objective, memory, tools)
        n += 1

        if finished:
            print(result)
            return

        if n > 2:
            print("Ended for reaching limit.")
            return

# What is ReAct reasoning and acting in language models?
startAgent()
