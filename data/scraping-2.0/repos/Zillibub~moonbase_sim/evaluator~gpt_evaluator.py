import openai
from evaluator.base_evaluator import BaseEvaluator


class GPTEvaluator(BaseEvaluator):

    def generate_code(self):

        case_prompt = self.load_prompt()

        codebase = self.load_codebase()

        prompt = f"""[PYTHON] {codebase} "[/PYTHON] 
        Generate code that will do {case_prompt}. The code must be executable. Always import all required modules.
        Always evaluate generated code. Do not write any tests. Always format generated code with ```python in the 
        beginning of the code block and with ``` in the end of generated code: 
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        completion = response.choices[0].message.content

        self.save_code(
            sequence=completion,
            output_path=self.output_path,
            code_stopwords=("```python", "```")
        )
