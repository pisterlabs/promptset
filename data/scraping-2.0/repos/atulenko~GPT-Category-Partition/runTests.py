
import openai
import argparse
import os

openai.api_key = os.environ["OPENAI_API_KEY"]


def generate_test_content(examples_filepath: str, tsl_filepath: str, output_filepath: str):
    """
        Calls the model 
    """

    with open(examples_filepath, "r") as examples_file:
        examples = examples_file.read()

    with open(tsl_filepath, "r") as tsl_file:
        tsl = tsl_file.read()

    tsl_array = tsl.split("\n\n")

    # I wish there was a session conecpt in GPT-3 so we didn't need to resend the training ex's each time.
    with open(output_filepath, "w") as output_file:
        for tsl_entry in tsl_array:
            prompt = examples + tsl_entry
            response = openai.Completion.create(
                engine="davinci",
                prompt=prompt,
                temperature=0,
                max_tokens=256,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["#end"]
            )

            output_file.write("\n#Running Test Case: \n")
            output_file.write(tsl_entry)
            output_file.write(response["choices"][0]["text"])
    
 
 
    # print(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--examplesFilepath', type=str, dest='examples_filepath', help='Path to file containing human created test cases', required=True)
    parser.add_argument('-t', '--tslFilepath', type=str, dest='tsl_filepath', help='Path to tsl file containing test cases definitions', required=True)
    parser.add_argument('-o', '--outputFilepath', type=str, dest='output_filepath', help='Path to file to write output', required=True)
    args = parser.parse_args()

    generate_test_content(args.examples_filepath, args.tsl_filepath, args.output_filepath)
