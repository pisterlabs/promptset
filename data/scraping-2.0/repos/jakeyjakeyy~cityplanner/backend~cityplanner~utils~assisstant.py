import openai

assistant = openai.beta.assistants.create(
    name="City Trip Planner",
    instructions="Users will give you information such as a city and general idea of their ideal night. Interpret their input and create a itinerary to visit different locations across their chosen city based on their interests.",
    # model="gpt-4-1106-preview",
    model="gpt-3.5-turbo",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "create_itinerary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state e.g. San Francisco, CA",
                        },
                        "locationTypes": {
                            "type": "string",
                            "description": "condense user requests into an ordered itinerary of each type, take liberties to add or adjust the list to make sure the list has at least 2 stops and create the perfect time out. Separate each stop with a , e.g. 'restaurant', 'amusement park', 'bar'",
                        },
                    },
                    "required": ["location", "locationTypes"],
                },
                "description": "Take user input and return one or more types of locations for the user to visit on their trip.",
            },
        }
    ],
)
