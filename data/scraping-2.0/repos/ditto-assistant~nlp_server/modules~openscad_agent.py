OPENSCAD_TEMPLATE = """
You are an OpenSCAD programmer. Your task is to take the user's prompt and code the response in a single OpenSCAD script in markdown format.
Do NOT assume any local files exist such as images or other files.

Examples:

User: openscad for a simple sphere.
Response:
```openscad
$fn = 100; // Set the resolution for the sphere

// Create a sphere
sphere(r = 20); // Adjust the radius as needed

User: <!prompt>
```


User: <!prompt>
Response:
"""

from langchain.chat_models import ChatOpenAI


class OpenSCADAgent:
    def __init__(self):
        self.template = OPENSCAD_TEMPLATE
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    def get_prompt_template(self, prompt):
        template = OPENSCAD_TEMPLATE
        template = template.replace("<!prompt>", prompt)
        return template

    def prompt(self, prompt):
        prompt = self.get_prompt_template(prompt=prompt)
        res = self.llm.call_as_llm(prompt)
        return res


if __name__ == "__main__":
    programmer_agent = OpenSCADAgent()
    res = programmer_agent.prompt(
        "Can you make me a computer mouse ergonomically CORRECT."
    )
    print(res)
