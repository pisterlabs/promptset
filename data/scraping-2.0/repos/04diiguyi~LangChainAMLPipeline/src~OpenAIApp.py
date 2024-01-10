import os
import argparse

from api_key import Az_OpenAI_api_key, Az_OpenAI_endpoint, Az_Open_Deployment_name_gpt35

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

import jsonlines

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to input data")
    parser.add_argument("--output", type=str, help="path to output data")
    args = parser.parse_args()

    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    os.environ["OPENAI_API_BASE"] = Az_OpenAI_endpoint
    os.environ["OPENAI_API_KEY"] = Az_OpenAI_api_key

    chat = AzureChatOpenAI(deployment_name=Az_Open_Deployment_name_gpt35,
                openai_api_version="2023-03-15-preview", temperature=0)

    print(args.input)

    with jsonlines.open(args.input) as reader:
        for obj in reader:
            print(obj)
            data = obj
            for test_data in data:
                print(test_data)
                messages = [
                    SystemMessage(content="You are a vehicle information extractor that extract 'Maker', 'Year', 'Model' and output in json format."),
                    HumanMessage(content=test_data["input"])
                ]
                response = chat(messages)

                test_data["actual"] = response.content

    with jsonlines.open(args.output, mode='w') as writer:
        writer.write(data)

if __name__ == "__main__":
    main()