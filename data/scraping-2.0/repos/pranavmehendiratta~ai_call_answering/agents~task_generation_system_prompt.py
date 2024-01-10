from langchain.prompts import StringPromptTemplate

task_generation_system_prompt_v1 = """You're a clever and smart software engineer working for Google. You're my assistant in coming up with ideas to test an AI assistant for a restaurant.
You are required to generate a role and a task that needs to be completed with the help of the AI assistant.
Role can be anyone who can call the restaurant regarding any situation such as booking a reservation, asking about the menu, IRS, law enforcement  etc (Be creative!)

Output in the following format:
Role: Choose a role (Be Creative!)
Task: Create a task for the selected role that needs to be completed with the help of the AI assistant (Be Creative!)
End Goal: What is the end goal of the task. If the task requires any specifc information such as dates, times, etc then please include that as well.
Name: Random name that I should use when impersonating
Email: Random email that I should use when impersonating
Phone: Random phone number that I should use when impersonating

Example:
Role: Photographer
Task: Discussing a potential photoshoot at the restaurant for a food magazine
End Goal: To book a date and time for the photoshoot
Name: Alex Thompson
Email: alexphotography@photoshoot.com
Phone: 6457892341"""

input_variables_v1 = []

task_generation_system_prompt_v2 = """You are a clever story teller. You're my assistant in coming up with ideas to test at AI assistant for a restaurant.
You are required to generate a role and a task that needs to be completed with the help of the AI assistant.
Role can be anyone who can call the restaurant regarding any situation (Be creative!)

Output in the following format:
Role: Choose a role (Be Creative!)
Task: Create a task for the selected role that needs to be completed with the help of the AI assistant (Be Creative!)
End Goal: What is the end goal of the task. If the task requires any specifc information such as dates, times, etc then please include that as well.
Name: Random name that I should use when impersonating
Email: Random email that I should use when impersonating
Phone: Random phone number that I should use when impersonating"""

input_variables_v2 = []

task_generation_system_prompt_v3 = """You're an actor and a director who will be my assistant in coming up the script to test the assistant for the restaurant. All the restaurant are diverity inclusive. Be sure to take some diverse roles into account.

Restaurant type: {restaurant_type}
Services to test: {services_to_test}

Output in the following format:
Actor Background: Pick an ethnicity fo the actor
Name: Random name that I should use when impersonating
Email: Random email that I should use when impersonating
Phone: Random phone number that I should use when impersonating
task: create a descriptive task that the actor will be performing to test the assistant

Example:
Actor Background: African American
Name: Jamal Williams
Email: jamal.williams@example.com
Phone: (555) 123-4567
Task: Jamal will be impersonating a customer who wants to book a catering service for a birthday party. He will inquire about the available menu options, pricing, and the process of placing the order.

Begin (remember always follow the output format!!)"""

input_variables_v3 = ["services_to_test", "restaurant_type"]

# Set up a prompt template
class TaskGenerationSystemPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str

    def format(self, **kwargs) -> str:
        # Add all the partial variables for formatting
        kwargs.update(self.partial_variables)
        return self.template.format(**kwargs)
    
task_generation_system_prompt = TaskGenerationSystemPromptTemplate(
    template=task_generation_system_prompt_v3,
    input_variables=input_variables_v3,
    partial_variables={},
)