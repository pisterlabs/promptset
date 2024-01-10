from pydantic import BaseModel, Field
from typing import Callable
from langchain.tools import Tool

# Define the tool schema
class ExceptionHandlerSchema(BaseModel):
    exception: Exception = Field(description="The exception instance to handle.")
    context: str = Field(description="The context or scenario where the exception occurred.")

# Define the tool function
def handle_exception(exception: Exception, context: str) -> str:
    # Log the exception (just printing for this example)
    print(f"Exception caught in context '{context}': {exception}")
    
    # Analyze the exception and provide a user-friendly message
    # (This is a basic example. In real scenarios, we might want to analyze the exception type, content, etc.)
    return f"An error occurred while {context}. Please check the inputs and try again."

# Define the tools for the agent
tools = [
    Tool(
        name="Exception Handler",
        func=handle_exception,
        args_schema=ExceptionHandlerSchema,
        description="Handles exceptions and provides user-friendly messages."
    ),
]

# Initialize the agent (assuming the same structure as PlanAndExecute)
exception_agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

# Example: Handling an exception
try:
    # Some code that might raise an exception
    result = 10 / 0
except Exception as e:
    user_message = exception_agent.run(f"- Handle the exception '{e}' that occurred while dividing numbers.")
    print(user_message)
