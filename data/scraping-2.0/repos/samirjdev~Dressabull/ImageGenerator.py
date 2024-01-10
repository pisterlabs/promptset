import os
import openai
import webbrowser
from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb




def rgb_to_color_name(rgb):

    # Convert the RGB list to a hexadecimal color representation
    hex_color = '#{:02X}{:02X}{:02X}'.format(*rgb)

    # Initialize variables to store the closest color and its distance
    closest_color = None
    closest_distance = float('inf')

    # Iterate through the CSS3_HEX_TO_NAMES dictionary to find the closest color
    for hex_value, color_name in CSS3_HEX_TO_NAMES.items():
        color_rgb = hex_to_rgb(hex_value)
        # Calculate the Euclidean distance between the colors
        distance = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
        if distance < closest_distance:
            closest_distance = distance
            closest_color = color_name

    return closest_color

def main(list_of_colors):

    colors = list_of_colors
    color_names = []

    # Example usage:
    for color in colors:
        color_name = rgb_to_color_name(color)
        #print(color_name)
        color_names.append(color_name)


    #print(f'RGB Color: {rgb_color}')


    #print(color_names[0])



    openai.api_key = ("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")


    response = openai.Image.create(
    prompt="Create an image of a unisex mannequin wearing a three-piece outfit consisting of a top, middle, and bottom. The top should be predominantly " + color_names[0] + ", the middle piece should be predominantly " +color_names[1] +" element, and the bottom should have predominantly " + color_names[2] + " base color. Ensure that the outfit is fashionable and visually appealing.",
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    print(image_url)