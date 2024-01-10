import openai

api_key = "YOUR_API_KEY"

project_name = "Project X"
project_description = "We are looking for a skilled developer to create a mobile app for our company. The app should have features like user registration, push notifications, and a user-friendly interface. Please provide your expertise and experience in your proposal."
project_budget = "$10,000 - $15,000"
project_deadline = "3 months"

proposal_prompt = f"Write a proposal for {project_name}. The project is about {project_description}. The budget for this project is {project_budget}, and the deadline is {project_deadline}."
response = openai.Completion.create(
    engine="davinci",
    prompt=proposal_prompt,
    max_tokens=150  # You can adjust the length of the generated text
)

proposal = response.choices[0].text

print(proposal)
