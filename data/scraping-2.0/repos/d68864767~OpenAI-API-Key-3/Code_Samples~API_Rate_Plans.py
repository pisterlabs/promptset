```python
# Import the OpenAI library
import openai

# Set your OpenAI API key
openai.api_key = 'your-api-key'

# Define a function to get the rate plan
def get_rate_plan():
    # Get the rate plan from the OpenAI API
    rate_plan = openai.RatePlan.retrieve()

    # Return the rate plan
    return rate_plan

# Define a function to set the rate plan
def set_rate_plan(plan_id):
    # Set the rate plan using the OpenAI API
    openai.RatePlan.update(id=plan_id)

# Define a function to list all available rate plans
def list_rate_plans():
    # Get the list of rate plans from the OpenAI API
    rate_plans = openai.RatePlan.list()

    # Return the list of rate plans
    return rate_plans

# Test the functions
print(get_rate_plan())
print(list_rate_plans())
```
