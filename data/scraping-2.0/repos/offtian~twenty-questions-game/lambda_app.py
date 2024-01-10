import json
import boto3
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from src.twenty_questions_game import TwentyQuestionsGame  # Import your game class

# Initialize outside of the lambda handler if you want to retain state across lambda invocations
llm = AzureChatOpenAI(...)  # Initialize with appropriate parameters
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("ChatHistoryTable")  # DynamoDB table to store chat history


def lambda_handler(event, context):
    # Extract user input from the API Gateway event
    user_input = json.loads(event["body"]).get("user_input")

    # Get the existing chat history from DynamoDB
    response = table.get_item(Key={"game_id": "unique_game_id"})
    chat_history = response.get("Item", {}).get("chat_history", [])

    # Initialize the memory with the chat history
    memory = ConversationBufferMemory(
        chat_memory=chat_history, memory_key="chat_history", return_messages=True
    )

    # Create a game instance with the language model and memory
    game = TwentyQuestionsGame(llm, memory)

    # Process the user input through the game logic
    game_response = game.run(user_input)

    # Update the chat history in DynamoDB
    new_chat_history = memory.chat_memory.messages()
    table.update_item(
        Key={"game_id": "unique_game_id"},
        UpdateExpression="SET chat_history = :val",
        ExpressionAttributeValues={":val": new_chat_history},
    )

    # Create the response object
    response_object = {
        "statusCode": 200,
        "body": json.dumps(
            {"game_response": game_response, "chat_history": new_chat_history}
        ),
        "headers": {"Content-Type": "application/json"},
    }

    # Return the response object
    return response_object
