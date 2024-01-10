from dataclasses import dataclass
import sys
from typing import Any
from typing_extensions import override

from typechat import Failure, Model, Result, Success, TypedDictTranslator, TypedDictValidator

import openai

from dotenv import dotenv_values
vals = dotenv_values()

@dataclass
class OpenAIModel(Model):
    model_name: str
    api_key: str

    @override
    def complete(self, input: str) -> Result[str]:
        try:
            ChatCompletion: Any = openai.ChatCompletion
            response = ChatCompletion.create(
                model=self.model_name,
                api_key=self.api_key,
                messages=[{"role": "user", "content": input}],
                temperature=0.0,
            )
            return Success(response.choices[0].message.content)
        except Exception as e:
            return Failure(str(e))

import coffee_api
with open(coffee_api.__file__, "r") as schema_file:
    api_schema = schema_file.read()

model = OpenAIModel(
    model_name=vals["OPENAI_MODEL"] or "",
    api_key=vals["OPENAI_API_KEY"] or ""
)

def main():
    validator = TypedDictValidator[coffee_api.Cart](api_schema, "Cart")
    translator = TypedDictTranslator(model, validator)
    print("☕> ", end="", flush=True)
    for line in sys.stdin:
        result = translator.translate(line)
        if isinstance(result, Failure):
            print("Translation Failed ❌")
            print(f"Context: {result.message}")
        else:
            result = result.value
            print("Translation Succeeded! ✅\n")
            print("JSON View")
            print(result)
        print("\n☕> ", end="", flush=True)


if __name__ == "__main__":
    print("Starting")
    main()
