import asyncio
import logging
import os

import quart
from langchain.llms import OpenAI

app = quart.Quart(__name__)

# Get the OpenAI API key from an environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

TEMPERATURES = [0.0, 0.5, 1.0, 1.5, 2.0]

LLMS = [
    OpenAI(openai_api_key=OPENAI_API_KEY, temperature=temperature)
    for temperature in TEMPERATURES
]

logging.basicConfig(level=logging.INFO)


@app.route("/", methods=["GET"])
async def index():
    return await quart.send_from_directory("backend/static", "index.html")


@app.route("/backup", methods=["GET", "POST"])
async def backup():
    # return index.html from templates if this is a get request
    if quart.request.method == "GET":
        return await quart.render_template("index.html")

    # unpack form data
    data = await quart.request.form
    prompt = data.get("prompt")
    if prompt is None:
        return await quart.render_template("index.html", error="No prompt provided")

    # Fetch 3 responses at different temperatures from OpenAi
    responses = await get_responses(prompt)

    # return index.html with the responses
    return await quart.render_template("index.html", prompt=prompt, responses=responses)


def get_single_response(llm, prompt):
    logging.info(f"Temp: {llm.temperature}, Prompt: {prompt}")
    response = llm(prompt)
    logging.info(f"Temp: {llm.temperature}, Response: {response}")
    return (llm.temperature, response)


async def get_responses(prompt):
    return await asyncio.gather(
        *(asyncio.to_thread(get_single_response, llm, prompt) for llm in LLMS)
    )


@app.route("/api", methods=["POST"])
async def api():
    # receive arguments as a josn object
    args = await quart.request.json

    # JSON object structure
    # {
    # "prompt": "This is a prompt",
    # }
    prompt = args.get("prompt")
    if prompt is None:
        return quart.jsonify(
            {"status": "error", "message": "No prompt provided", "responses": []}
        )

    # Fetch 3 responses at different temperatures from OpenAi
    # and return them as a JSON object
    responses = await get_responses(prompt)

    # return a json object
    return quart.jsonify(
        {
            "status": "success",
            "responses": [
                {"temperature": temperature, "response": response}
                for temperature, response in responses
            ],
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=8080)
