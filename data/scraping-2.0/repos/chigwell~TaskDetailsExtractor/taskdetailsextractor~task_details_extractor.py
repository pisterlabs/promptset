import openai

class TaskDetailsExtractor:
    def __init__(self, openai_api_key, model="gpt-3.5-turbo", max_tokens=1500):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        self.max_tokens = max_tokens

    def analyze_task_details(self, project_structure, project_languages, task_description):
        # Build the prompt for GPT-3
        system_message = self._build_system_message(project_structure, project_languages)
        user_message = self._build_user_message(task_description)

        # Query GPT-3
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=self.max_tokens
        )

        # Extract the response details
        return response.choices[0].message.content

    def _build_system_message(self, project_structure, project_languages):
        # Construct a system message with project details and strict response format rules
        return f"""The system will analyze the given task description based on the project's file structure and technology stack. Provide a list of detailed actions or files necessary for completing the task based on the provided project structure and languages. The response must strictly follow the JSON structure outlined below:
        Project Structure:
        {project_structure}

        Project Languages:
        {project_languages}

        Strict format for response: 
        A JSON array of objects each representing a task with the following structure:
        [
            {{
                "step": "1",
                "type_action": "c",  // c - create, r - read, u - update, d - delete
                "file_path": "relative/path/to/file",
            }},
            //...  More tasks if necessary
        ]
        Each action should be a discrete operation on a file necessary to complete the given task."""

    def _build_user_message(self, task_description):
        # Construct the user message with the task description
        return f"Please assess the feasibility of the following task based on the project's details provided above:\n\nTask Description: {task_description}\n\nRespond with a detailed list of actions in JSON format as instructed."

# Usage example
# extractor = TaskDetailsExtractor('your-openai-api-key')
# task_details = extractor.analyze_task_details(project_structure, project_languages, "Refactor the database schema to improve performance.")
# print(task_details)
