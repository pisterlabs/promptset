from flask import Flask, render_template, request
import os
import pprint
import re
import openai
from dotenv import load_dotenv
from flask import jsonify
import time
import random


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

application = Flask(__name__,
                    template_folder='templates',
                    static_url_path='',
                    static_folder='static'
                    )
application.config["ENV"] = "production"
application.config["DEBUG"] = False


HEX_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")


def generate_palette(msg):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a color palette generating assistant that helps users find the perfect color palette for their desired theme or design. Your answers should be grounded in the principles of color theory, ensuring harmonious and complementary color combinations. Consider aspects such as triadic schemes and analogous colors when providing suggestions. Suggest only one color scheme, palette, or theme per message. Only provide hexadecimal color codes for each color suggestion in your response. Do not provide any other information or content in your response",
            },
            {"role": "user", "content": msg},
        ],
    )

    message_content = completion.choices[0].message["content"]

    # Using the pre-compiled pattern
    palette = HEX_COLOR_PATTERN.findall(message_content)

    return palette


@application.route("/palette", methods=["POST"])
def palette_prompt():

    # application.logger.info("Post request received!")

    query = request.form.get("query")

    # application.logger.info(f"Generating palette for query: {query}")
    colors = generate_palette(query)

    return jsonify({"colors": colors})


@application.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    application.run()


# def random_color():

#     # Generate a random hex color code.

#     return "#{:06x}".format(random.randint(0, 0xFFFFFF))


# def is_valid_hex_color(color):

#     # Check if the provided string is a valid hex color code.

#     return bool(re.match(r"^#(?:[0-9a-fA-F]{3}){1,2}$", color))


# def complementary_color(hex_color):

#     # Compute the complementary color of the given hex color code.

#     # Ensure the color is valid
#     if not is_valid_hex_color(hex_color):
#         raise ValueError("Invalid hex color code.")

#     # Convert the hex code to RGB values
#     r = int(hex_color[1:3], 16)
#     g = int(hex_color[3:5], 16)
#     b = int(hex_color[5:7], 16)

#     comp_r = 255 - r
#     comp_g = 255 - g
#     comp_b = 255 - b

#     return "#{:02X}{:02X}{:02X}".format(comp_r, comp_g, comp_b)


# @application.route("/random_color", methods=["GET"])
# def random_color_route():

#     # Generate and return a random color code.

#     color = random_color()
#     return jsonify({"color": color})


# @application.route("/complementary", methods=["POST"])
# def complementary_color_route():

#     # compute and return complementary color.

#     data = request.get_json()
#     hex_color = data.get("color")
#     try:
#         complementary = complementary_color(hex_color)
#         return jsonify({"complementary": complementary})
#     except ValueError:
#         return jsonify({"error": "Invalid color provided."}), 400
