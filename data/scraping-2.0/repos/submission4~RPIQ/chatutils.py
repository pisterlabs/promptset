import json
from flask import jsonify
from openai import OpenAI
import os
from settings.ignoredsettings import OPENAIAPIKEY

print("openaikey", OPENAIAPIKEY)
OPENAIAPIKEY = "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"
# os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAIAPIKEY
apicheck = os.environ["OPENAI_API_KEY"]
client = OpenAI()
print(client.api_key)
# set the api key
client.api_key = os.environ["OPENAI_API_KEY"]
print(client.api_key)
print(apicheck)
print(OPENAIAPIKEY)
print(os.environ["OPENAI_API_KEY"])
print(client.models.list)


def checkAPIexsits():
    api_key = os.environ["OPENAI_API_KEY"]
    try:
        if api_key == None:
            print("api key not set")

        elif apicheck == None:
            print("api key not set")

        print("api key exists")
        return

    except Exception or apicheck == None or api_key == None:
        print("api key needs to be set")
        print(str(Exception))


checkAPIexsits()


def get_ai_message(user_input, history):
    # take the prompt and generate the next text
    print("history", history)
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "MAKE SURE TO ALWAYS STAY IN CHARACTER and only respond as your role"
                + str(history)
                + "PLEASE START INTERACTION GIVING YOUR ROLE AND GREET THE OTHER PERSON if youve already greeted them dont do it twice",
            },
            {"role": "user", "content": str(user_input)},
        ],
    )

    respon = completion.choices[0].message.content
    return respon


def checkWinCondition(history, win1, win2):
    # take the prompt and generate the next text
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "Evaluate the user interaction and determine the winner in a JSON format. The response should include metrics scores and feedback structured as a JSON object."
                + """
            {
            "EvaluationTask": "Evaluate user interaction",
            "Metrics": {
                "Relevance": "Rate how the interaction aligns with the original prompt (1-5)",
                "Clarity": "Assess the clarity of the user's instructions or questions (1-5)",
                "Completeness": "Evaluate if all necessary details are provided (1-5)",
                "Logic": "Gauge the logic behind the user's statements (1-5)",
                "Creativity": "Rate the user's creativity and exploration (1-5)",
                "Engagement": "Assess user engagement with model responses (1-5)",
                "Politeness": "Evaluate the politeness and tone of the interaction (1-5)",
                "Adaptability": "Determine adaptability to model's outputs (1-5)",
                "Language": "Check appropriateness of the language used (1-5)",
                "Effectiveness": "Give an overall effectiveness rating (1-5)"
            },
            "AnalysisOfWinConditions": "Review user's interaction history to assess win conditions",
            "WinAssessment": {
                "Winner": "Name or description of the winner, if applicable",
                "ReasonForWin": "Explanation of what led to the win",
                "PercentageMetrics": {
                    "Role1": "Percentage success rate for Role 1",
                    "Role2": "Percentage success rate for Role 2",
                }
            },
            "ResponseFormat": "Provide scores and brief feedback for each metric, determine the winner, and explain the reason for the win with percentage metrics for each role, formatted as a JSON object"
            }""",
            },
            {"role": "system", "content": "always return the JSON object that is specified"},
            {"role": "user", "content": str(history)},
        ],
    )

    # Extracting the response
    respon = completion.choices[0].message.content

    # Check if the response is already in JSON format
    try:
        # if respon begins with ``` and ends with ```, it is already in JSON format
        # so we remove the ``` and ``` and return the response as a json object
        if respon.startswith("```") and respon.endswith("```"):
            respon = respon[3:-3]
            respon_dict = json.loads(respon)
            print("printing json response")
            print(type(respon_dict))
            return jsonify(json.loads(respon))
        else:
            # If unsuccessful, return the response as a string
            print("printing string response")
            return respon

    except Exception as e:
        # If unsuccessful, return the response as a string
        print(e, "error")
        return respon


fakehistory = [
    {
        "category": "Beginning Test RPs",
        "title": "The Negotiation Challenge",
        "roles": [{"Buyer": "Alex Mitchell"}, {"Seller": "Emily Davis"}],
        "objectives": [
            {"Alex Mitchell (Buyer)": "Negotiate a favorable deal for a purchase."},
            {
                "Emily Davis (Seller)": "Secure a profitable agreement while meeting buyer's needs."
            },
        ],
        "win_conditions": [
            {
                "Alex Mitchell (Buyer)": "Successfully negotiate a deal that meets your needs and expectations."
            },
            {
                "Emily Davis (Seller)": "Secure a profitable agreement and ensure Alex is satisfied with the deal."
            },
        ],
        "scenario": "# Scenario\nAlex Mitchell is a buyer looking to make a purchase, and Emily Davis is the seller. Both must engage in a negotiation to reach a mutually satisfactory agreement. Navigate the negotiation process, discuss terms, and work towards a deal that benefits both parties.\n- Alex Mitchell (Buyer): Clearly communicate your needs and negotiate for a favorable deal.\n- Emily Davis (Seller): Work to secure a profitable agreement while meeting the buyer's requirements.",
        "user_role": "Buyer Alex Mitchell",
        "ai_role": {"Buyer": "Alex Mitchell"},
    },
    "user: hi",
    "ai: Hello Alex, I'm Emily Davis. It's a pleasure to meet you. What can I do for you today?",
    "user: I'm interested in purchasing your product, but I'm hoping we can discuss the price.",
    "ai: Great to hear from you! I'm sure we can find a price that works for both of us. What were you thinking?",
    "user: I've done some market research, and I believe a fair price would be around $20,000. What are your thoughts?",
    "ai: I appreciate your offer. However, considering the quality and features of our product, I was thinking more along the lines of $25,000.",
    "user: I understand your point, but my budget is a bit tight. Could we meet in the middle at $22,500?",
    "ai: That's a reasonable compromise. How about we include an extended warranty at that price to add value to your purchase?",
    "user: That sounds like a good deal. I'm happy with $22,500 plus the extended warranty. Let's proceed with this.",
    "ai: Fantastic! I'll prepare the necessary paperwork. It's been a pleasure negotiating with you, Alex.",
    "user: Same here, Emily. Looking forward to doing more business with you in the future.",
]

winCheckdata = {
    "EvaluationTask": "Evaluate user interaction",
    "Metrics": {
        "Relevance": "Rate how the interaction aligns with the original prompt (1-5)",
        "Clarity": "Assess the clarity of the user's instructions or questions (1-5)",
        "Completeness": "Evaluate if all necessary details are provided (1-5)",
        "Logic": "Gauge the logic behind the user's statements (1-5)",
        "Creativity": "Rate the user's creativity and exploration (1-5)",
        "Engagement": "Assess user engagement with model responses (1-5)",
        "Politeness": "Evaluate the politeness and tone of the interaction (1-5)",
        "Adaptability": "Determine adaptability to model's outputs (1-5)",
        "Language": "Check appropriateness of the language used (1-5)",
        "Effectiveness": "Give an overall effectiveness rating (1-5)",
    },
    "AnalysisOfWinConditions": "Review user's interaction history to assess win conditions",
    "WinAssessment": {
        "Winner": "Name or description of the winner, if applicable",
        "ReasonForWin": "Explanation of what led to the win",
        "PercentageMetrics": {
            "Role1": "Percentage success rate for Role 1",
            "Role2": "Percentage success rate for Role 2",
        },
    },
    "ResponseFormat": "Provide scores and brief feedback for each metric, determine the winner, and explain the reason for the win with percentage metrics for each role, formatted as a JSON object",
}
