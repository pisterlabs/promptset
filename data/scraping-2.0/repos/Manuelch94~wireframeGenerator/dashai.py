import os
import argparse
import openai
import json

# Set your API as describe in the README
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(prompt):
    
    # You can change the generator, this generator is a single line chart using the grid layout
    dashboard_generator = '''
    {"visualizations":{"viz_LYCo0hNI":{"type":"splunk.line","title":"line chart"}},"dataSources":{},"defaults":{"dataSources":{"ds.search":{"options":{"queryParameters":{"latest":"$global_time.latest$","earliest":"$global_time.earliest$"}}}}},"inputs":{"input_global_trp":{"type":"input.timerange","options":{"token":"global_time","defaultValue":"-24h@h,now"},"title":"Global Time Range"}},"layout":{"type":"absolute","options":{"display":"auto-scale","height":2000,"width":2000},"structure":[{"item":"viz_LYCo0hNI","type":"block","position":{"x":20,"y":20,"w":300,"h":300}}],"globalInputs":["input_global_trp"]},"description":"","title":"mydashboard"}
    '''
    example = json.dumps(dashboard_generator)
    
    
    complete_prompt = f"""
    Act as a expert building dashboard using dashboard studio in splunk.
    Understand the following JSON schema, you will use the schema only as a guide. 
    The only sections of the schema you are allow to edit are: \"layout\" and \"visualizations\".
    Only respond with the JSON object that satifies the request only after each request, no text before or after your response.
    Every vizualization always follows this format, this is an example: {example}
    User Request: {prompt}
    """

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=complete_prompt,
        temperature=0.41,
        max_tokens=2880,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].text.strip()

def main():
    parser = argparse.ArgumentParser(description="Dashboard Studio Wireframe Generator Tool")
    parser.add_argument("request", type=str, help="User request for the wireframe")

    args = parser.parse_args()
    user_request = args.request

    generated_response = generate_response(user_request)
    print(generated_response)

if __name__ == '__main__':
    main()
