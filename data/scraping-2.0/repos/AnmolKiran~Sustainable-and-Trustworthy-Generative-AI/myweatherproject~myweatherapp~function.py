from openai import OpenAI
import json
from googlesearch import search
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

client = OpenAI()


def perform_search(query):
    # Use the Google Search API or web scraping logic here
    results = []
    for j in search(query, num=5, stop=5, pause=2):  # Fetching the top 5 search results
        results.append({"title": "Result", "link": j, "snippet": "Description of the result."})
    return results

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # 2 sentences in the summary
    return ' '.join(str(sentence) for sentence in summary)

def search_function(query):
    # Your search logic here
    results = perform_search(query)

    # Summarize each search result
    for result in results:
        result['summary'] = summarize_text(result['snippet'])

    return json.dumps({"results": results})







def chat_function(message):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": message}]
    
    # Step 2: make an API call to OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        
        messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
       # tool_choice="auto",  # auto is default, but we'll be explicit
    )
    
    # Step 3: Extract and return the assistant's reply
    response_message = response.choices[0].message
    return json.dumps({"response": response_message.content})




    



def trusted_sources_function(topic):
    # Your trusted sources logic here
    sources = get_trusted_sources(topic)
    return json.dumps({"sources": sources})

def single_value_dashboard_function(metric):
    # Your single value dashboard logic here
    value = get_single_value_metric(metric)
    return json.dumps({"value": value})

def chart_dashboard_function(chart_type, data):
    # Your chart dashboard logic here
    chart_data = generate_chart(chart_type, data)
    return json.dumps({"chart_data": chart_data})


def news_dashboard_function(user_query, current_date, news_headlines):
    # Generate a prompt using user query, current date, and news headlines
    prompt = f"These are the latest news headlines regarding '{user_query}' on {current_date}. Give me a brief 1 paragraph summary for '{user_query}'.\nStart:\n"

    for news_entry in news_headlines:
        prompt += f"\n{news_entry['source']}\n{news_entry['headline']}\n{news_entry['timestamp']}\n"
  

    # Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    # Extract and return the summary from the OpenAI response
    response_message = response.choices[0].message
    summary = response_message.content.strip()
    
    return json.dumps({"summary": summary})

# Example usage:
user_query = "steel price forecast"
current_date = "11th November 2023"
news_headlines = [
    {"source": "CNBC", "headline": "Goldman Sachs sees 'clear deficit' of iron ore", "timestamp": "Yesterday"},
    # Add more news entries as needed
]

result = news_dashboard_function(user_query, current_date, news_headlines)
print(result)



def patents_search_function(query):
    # Your patents search logic here (replace with actual implementation)
    patents_data = [
        {
            "title": "舜平 山崎 株式会社半導体エネルギー研究所",
            "priority": "2003-06-16",
            "filed": "2023-08-10",
            "granted": "2023-09-29",
            "published": "2023-09-29",
            "description": "An object of the present invention is to form an auxiliary electrode that can prevent a luminance gradient caused by a potential drop of a counter electrode from becoming visible even as the definition of a light emitting device progresses...",
        },
        {
            "title": "シャシュア，アムノン モービルアイ ビジョン テクノロジーズ リミテッド",
            "priority": "2018-03-20",
            "filed": "2023-07-11",
            "granted": "2023-10-23",
            "published": "2023-10-23",
            "description": "In the processor circuit, obtaining a planned driving behavior to achieve a navigation goal of the host vehicle; receiving sensor data from a sensor device representative of the environment surrounding the host vehicle; identifying a target vehicle moving within the environment...",
        },
        # Add more patent data as needed
    ]

    # Summarize the patents related to the user query
    summary_paragraph = f"These are the latest patents related to '{query}'. Summarize these patents in specific in 1 paragraph and relate them to '{query}'.\n\n"
    for patent in patents_data:
        summary_paragraph += f"{patent['title']} ({patent['priority']}): {patent['description']}\n\n"

    # Return the summary as a JSON string
    return json.dumps({"summary": summary_paragraph})



# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
      #  tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "chat_function": chat_function,
            "search_function": search_function,
            "trusted_sources_function": trusted_sources_function,
            "single_value_dashboard_function": single_value_dashboard_function,
            "chart_dashboard_function": chart_dashboard_function,
            "news_dashboard_function": news_dashboard_function,
            "patents_search_function": patents_search_function,
            "get_current_weather": get_current_weather,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.json()
print(run_conversation())