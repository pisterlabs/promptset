import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from flask import Flask, request, jsonify
from datetime import datetime
import json

kernel = sk.Kernel()

# %%
# OpenAI API
# api_key, org_id = sk.openai_settings_from_dot_env()
# kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key, org_id))

# Azure OpenAI API
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat_completion", AzureChatCompletion(deployment, endpoint, api_key)
)

def analysze_news(news_list, Events_list):

    # Import Semantic Skill
    NewsAnalysisSkill = kernel.import_semantic_skill_from_directory("../samples/skills", "NewsAnalysisSkill")
    
    # Define Semantic Functions
    news_classifier_function = NewsAnalysisSkill["NewsClassifier"]
    news_summary_function = NewsAnalysisSkill["EventNewsSummary"]
    news_extend_summary_function = NewsAnalysisSkill["EventsNewsSummaryExtend"]
    news_properties_function = NewsAnalysisSkill["EventProperties"]

    context = kernel.create_new_context()

    Events_dict = {}

    context["Events"] = Events_list

    for i in range(len(news_list["articles"]["results"])):

        news = str(news_list["articles"]["results"][i]) #["articles"]["results"]

        print(f"News: {news}")

        context["News"] = news

        news_classification = news_classifier_function(context=context)
        event = news_classification.variables.input
        # Show the response
        print(f"Event: {news_classification}")
        
        news_properties = news_properties_function(context=context)
        properties = json.loads(news_properties.variables.input)
        print(f"Properties: {news_properties}")
        

        if "ErrorCodes.ServiceError" in event:
            print("Error: ", event)
        else:
            # Append the new interaction to the chat history
            context["Events"] += f"- {event}\n"

            if event in Events_dict:
                Events_dict[event]["News"].append(news)
                for prop in properties:
                    if prop not in Events_dict[event]["Properties"]:
                        Events_dict[event]["Properties"].append(prop)
                    else:
                        pass
            else:
                Events_dict.update({event: {"News": [news], "Properties": properties}})

            if "Summary" in Events_dict[event]:
                context["Old_Summary"] = Events_dict[event]["Summary"]
                context["New_Event_News_List"] = news
                Events_dict[event]["Summary"] = news_extend_summary_function(
                    context=context
                ).variables.input

            else:
                context["Event_News_List"] = news
                Events_dict[event]["Summary"] = news_summary_function(
                    context=context
                ).variables.input

    return Events_dict, context["Events"]

# %%



app = Flask(__name__)
Events_list = '' 

@app.route('/news_analysis', methods=['GET', 'POST'])
def news_analysis():
    try:
        print(json.dumps(newsData, indent=4))  # Pretty print the incoming data
        # Check if newsData is not empty
        if newsData and len(newsData) > 0:
            print(newsData[0].title)  # Print the title of the first article
        return {"message": "Processed successfully"}
        
        # Triggering the function
        #analyzed_news_dict, Events_list = analysze_news(news_list, Events_list)

        #now = datetime.now()

        #json_results_path = ''
        
        # Saving the results to a JSON file
        #with open(f'{json_results_path}/analyzed_news_{now.strftime("%H-%M-%S_%d-%m-%Y")}.json', 'w') as f:
        #    json.dump(analyzed_news_dict, f)

        # Sending a success response
        #return jsonify({"status": "success", "message": "Data processed successfully"}), 200

    except Exception as e:
        # Handling errors and sending an error response
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/')
def index():
    return "Welcome to News Pathfinder!"


if __name__ == "__main__":
    # Running the Flask app on port 5000
    app.run(port=5000)