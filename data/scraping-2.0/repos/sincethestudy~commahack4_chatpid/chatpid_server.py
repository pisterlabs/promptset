import random
from PIL import Image
import openai
import json
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import matplotlib.patches as patches
import modal
from PIL import Image
import numpy as np
from collections import deque
# from elevenlabs import generate, play

# audio = generate(
#   text="A dollar was added to your A100 bill. Two dollars have been added to your OpenAI Bill",
#   voice="Bella",
#   model="eleven_multilingual_v2"
# )




history = []

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks, save_path=None):
    # plt.imshow(np.array(raw_image))
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    # Calculate total area of the image
    total_area = raw_image.size[0] * raw_image.size[1]

    labels_list = []
    label_images_list = []

    for mask in masks:
        # Show the mask with random color
        # show_mask(mask, ax=ax, random_color=True)

        # Label the mask
        label_image = label(mask)
        label_images_list.append(label_image)

        # Get properties of the labeled regions
        regions = regionprops(label_image)

        # Classify regions as floor, wall or neither based on their position, size and edge contact
        labels = np.zeros_like(label_image, dtype=np.uint8)
        for region in regions:
            # Check if region is large (more than 10% of total area)
            if region.area > total_area * 0.1:
                # Check if region has more pixels touching the bottom, top, left or right edge
                bottom_edge_touch = np.sum(region.coords[:, 0] == mask.shape[0] - 1)
                top_edge_touch = np.sum(region.coords[:, 0] == 0)
                left_edge_touch = np.sum(region.coords[:, 1] == 0)
                right_edge_touch = np.sum(region.coords[:, 1] == mask.shape[1] - 1)
                # If the region has more pixels touching the bottom edge than the other edges, it's a floor
                if bottom_edge_touch > max(top_edge_touch, left_edge_touch, right_edge_touch):
                    labels[label_image == region.label] = 1  # Floor
                else:
                    labels[label_image == region.label] = 2  # Wall
            else:
                labels[label_image == region.label] = 3  # Neither

        labels_list.append(labels)

        # Add text labels
        # for region in regions:
        #     # Position text at the centroid of the region
        #     y, x = region.centroid
        #     if labels[int(y), int(x)] == 1:
        #         ax.text(x, y, 'Floor', fontsize=12, color='red')
        #     elif labels[int(y), int(x)] == 2:
        #         ax.text(x, y, 'Wall', fontsize=12, color='red')
        #     elif labels[int(y), int(x)] == 3:
        #         # ax.text(x, y, 'Neither', fontsize=12, color='red')
        #         continue

    # plt.axis("off")
    # if save_path is not None:
    #     plt.savefig(save_path)
    # plt.show()
    # del mask
    gc.collect()

    return labels_list, label_images_list

def image_to_ascii(labels, label_image, ascii_width=50, ascii_height=30):
    # Initialize an empty ASCII image
    ascii_image = [['Wall']*ascii_width for _ in range(ascii_height)]

    # Calculate the step size for sampling the image
    step_size_x = max(label_image.shape[1] // ascii_width, 1)
    step_size_y = max(label_image.shape[0] // ascii_height, 1)



    # Plot the labels array
    # fig, ax = plt.subplots(figsize=(6, 6))
    # ax.imshow(labels, cmap='jet')
    # plt.title('Labels before ASCII conversion')
            
    for y in range(0, label_image.shape[0], step_size_y):
        for x in range(0, label_image.shape[1], step_size_x):
            # Calculate the histogram of labels within the chunk
            chunk_labels = labels[y:y+step_size_y, x:x+step_size_x]
            label_counts = np.bincount(chunk_labels.flatten())

            # Find the most common label
            most_common_label = np.argmax(label_counts)

            # Assign the corresponding ASCII character based on the most common label
            ascii_y = min(y // step_size_y, ascii_height - 1)
            ascii_x = min(x // step_size_x, ascii_width - 1)
            if most_common_label == 1:
                ascii_image[ascii_y][ascii_x] = 'Floor'
                color = 'green'  # Green for floor
            elif most_common_label == 2:
                ascii_image[ascii_y][ascii_x] = 'Wall'
                color = 'red'  # Red for wall
            else:
                ascii_image[ascii_y][ascii_x] = 'Wall'
                color = 'blue'  # Blue for none

            # Draw rectangle for the chunk on the image
            # rect = patches.Rectangle((x, y), step_size_x, step_size_y, linewidth=1, edgecolor='r', facecolor=color, alpha=1)
            # ax.add_patch(rect)

    # plt.show()

    label = "CURRENT LOCATION"
    center_column = ascii_width // 2
    ascii_image[-1][center_column] = label

    # Define the labels
    labels = {
        "TOP LEFT": (0, 0),
        "TOP RIGHT": (0, -1),
        "BOTTOM LEFT": (-1, 0),
        "BOTTOM RIGHT": (-1, -1),
    }

    # Place the labels in the ascii_image
    for label, position in labels.items():
        # Check if the label fits in one cell
        # Replace the specific cell of the ASCII image with the label
        ascii_image[position[0]][position[1]] = label


    # Convert the ASCII image to a string
    ascii_str = '\n'.join(' '.join(row) for row in ascii_image)

        # Convert the ASCII image to a numerical format for plotting
    ascii_numerical = [[0 if cell == 'Wall' else 1 if cell == 'Floor' else 2 for cell in row] for row in ascii_image]

    # Create a color map for the plot
    cmap = plt.get_cmap('viridis', 3)  # 3 distinct colors for 'None', 'Floor', and 'Wall'

    # Create the plot
    # plt.figure(figsize=(10, 10))
    # plt.imshow(ascii_numerical, cmap=cmap)

    # Add a color bar
    # cbar = plt.colorbar(ticks=[0, 1, 2])
    # cbar.ax.set_yticklabels(['None', 'Floor', 'Wall'])

    # Display the plot
    # plt.show()

    with open('./log/ascii_log.txt', 'a') as f:
        f.write(ascii_str + '\n\n')



    return ascii_str, ascii_image

def describe_direction(start_row, start_col, row_increment, col_increment, map_2d):
    """Describe the tiles in a specific direction from a starting point."""
    description = []
    row, col = start_row, start_col

    print(row, col, row_increment, col_increment, map_2d)
    
    while 0 <= row < len(map_2d) and 0 <= col < len(map_2d[0]):
        if map_2d[row][col] == "Wall":
            description.append("a wall")
            break
        elif map_2d[row][col] == "Floor":
            description.append("floor")
        row += row_increment
        col += col_increment
    
    # Condensing consecutive similar descriptions
    condensed_description = []
    prev_tile = None
    count = 0
    for tile in description:
        if tile == prev_tile:
            count += 1
        else:
            if prev_tile and count > 1:
                condensed_description.append(f"{count} tiles of {prev_tile}")
            elif prev_tile:
                condensed_description.append(prev_tile)
            count = 1
        prev_tile = tile
    if count > 1:
        condensed_description.append(f"{count} tiles of {prev_tile}")
    elif prev_tile:
        condensed_description.append(prev_tile)
    
    return ", followed by ".join(condensed_description)

def describe_surroundings(map_2d):
    """Describe the surroundings based on the current location in the map."""
    
    # Finding the current location (marked as "CENTER") in the map
    current_location = None
    for i, row in enumerate(map_2d):
        for j, tile in enumerate(row):
            if tile == "CURRENT LOCATION":
                current_location = (i, j)
                break
        if current_location:
            break

    print(current_location)
    
    # Describing the surroundings
    front_description = describe_direction(current_location[0]-1, current_location[1], -1, 0, map_2d)
    left_description = describe_direction(current_location[0], current_location[1]-1, 0, -1, map_2d)
    right_description = describe_direction(current_location[0], current_location[1]+1, 0, 1, map_2d)
    
    # Creating the final description
    final_description = f"""
    Directly in front of you: {front_description}.
    To your left: {left_description}.
    To your right: {right_description}.
    """
    
    return final_description.strip()

def describe_orientation(coord, current_location):
    """Describes the orientation of a coordinate relative to the 'CURRENT LOCATION'."""
    x_diff = coord[0] - current_location[0]
    y_diff = coord[1] - current_location[1]
    
    if x_diff < 0 and y_diff == 0:
        return "W"  # North
    if x_diff > 0 and y_diff == 0:
        return "S"  # South
    if x_diff == 0 and y_diff < 0:
        return "A"  # West
    if x_diff == 0 and y_diff > 0:
        return "D"  # East
    if x_diff < 0 and y_diff < 0:
        return "W-A"  # North-West
    if x_diff < 0 and y_diff > 0:
        return "W-D"  # North-East
    if x_diff > 0 and y_diff < 0:
        return "S-A"  # South-West
    if x_diff > 0 and y_diff > 0:
        return "S-D"  # South-East
    return "at the 'CURRENT LOCATION'"

def describe_size(cluster_size, total_floor_cells):
    """Describes the size context of a cluster."""
    if cluster_size < (0.25 * total_floor_cells):
        return "a small"
    if cluster_size < (0.5 * total_floor_cells):
        return "a medium-sized"
    return "a large"

def is_at_boundary(cluster, matrix):
    """Checks if a cluster has any cells at the boundary of the map."""
    for x, y in cluster:
        if x == 0 or y == 0 or x == len(matrix) - 1 or y == len(matrix[0]) - 1:
            return True
    return False

def get_neighbors(x, y, matrix):
    """Returns valid neighbors for a given cell in the matrix."""
    neighbors = []
    for i, j in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
        if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]) and matrix[i][j] == "Floor":
            neighbors.append((i, j))
    return neighbors

def is_walled_area(cluster, matrix):
    """Check if a cluster of 'Floor' cells is completely surrounded by 'Wall' cells."""
    for x, y in cluster:
        neighbors = get_neighbors(x, y, matrix)
        for nx, ny in neighbors:
            if (nx, ny) not in cluster:
                return False
    return True

def generate_detailed_map_report_v2(cleaned_matrix):
    def find_clusters_v2(matrix):
        visited = set()
        clusters = []

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if (i, j) not in visited and matrix[i][j] == "Floor":
                    # Start of a new cluster
                    cluster = []
                    queue = deque([(i, j)])
                    while queue:
                        x, y = queue.popleft()
                        if (x, y) not in visited:
                            visited.add((x, y))
                            cluster.append((x, y))
                            neighbors = get_neighbors(x, y, matrix)
                            queue.extend(neighbors)
                    clusters.append(cluster)

        return clusters
    """Generates a super detailed report based on the input map string."""
    
    # Convert the string into a 2D list representation
    # rows = [row.split() for row in map_string.strip().split("\n")]

    # # Cleaning the matrix to remove labels and create a uniform matrix
    # cleaned_matrix = []
    # for row in rows:
    #     cleaned_row = [cell for cell in row if cell in ["Wall", "Floor"]]
    #     cleaned_matrix.append(cleaned_row)

    print(cleaned_matrix)

    # Finding the current location (marked as "CENTER") in the map
    current_location = None
    for i, row in enumerate(cleaned_matrix):
        for j, tile in enumerate(row):
            if tile == "CURRENT LOCATION":
                current_location = (i, j)
                break
        if current_location:
            break

    # General Overview
    overview = f"The scene depicts a structured area of dimensions {len(cleaned_matrix)}x{max(len(row) for row in cleaned_matrix)}. Your current position is marked at coordinates {current_location}.\n"

    # Composition
    wall_count = sum(row.count("Wall") for row in cleaned_matrix)
    floor_count = sum(row.count("Floor") for row in cleaned_matrix)
    composition = f"The area is divided into {wall_count} obstructions (or 'Wall' cells) and {floor_count} open spaces (or 'Floor' cells).\n"

    # Pathways
    clusters = find_clusters_v2(cleaned_matrix)
    pathways_report = f"There are {len(clusters)} distinct open areas or pathways. "
    pathways_details = []
    for cluster in clusters:
        start_orientation = describe_orientation(cluster[0], current_location)
        size_description = describe_size(len(cluster), floor_count)
        if is_at_boundary(cluster, cleaned_matrix):
            pathways_details.append(f"{size_description} pathway starting {start_orientation} leads to the boundary of the area, suggesting a potential exit or entrance.")
        else:
            pathways_details.append(f"{size_description} pathway starts {start_orientation}.")
    pathways_description = ' '.join(pathways_details)

    # Walled Areas
    walled_areas = [cluster for cluster in clusters if is_walled_area(cluster, cleaned_matrix)]
    walled_areas_report = f"There are {len(walled_areas)} completely enclosed or walled areas. "
    walled_areas_details = []
    for area in walled_areas:
        start_orientation = describe_orientation(area[0], current_location)
        size_description = describe_size(len(area), floor_count)
        walled_areas_details.append(f"{size_description} enclosed area starts {start_orientation}.")
    walled_areas_description = ' '.join(walled_areas_details)

    return overview + composition + pathways_report + pathways_description + walled_areas_report + walled_areas_description


def process_image_and_generate_direction(random_image, image_mask, show_masks_on_image):

    random_image = Image.fromarray(random_image)

    # Apply image processing
    aspect_ratio = random_image.size[0] / random_image.size[1]
    new_height = 100
    new_width = int(new_height * aspect_ratio)
    random_image = random_image.resize((new_width, new_height), Image.LANCZOS)

    width, height = random_image.size
    left = width * 0.2
    right = width * 0.8
    top = height * 0.05
    bottom = height * 0.95
    random_image = random_image.crop((left, top, right, bottom))

    new_w = random_image.size[0] 
    new_h = random_image.size[1]

    # Update labels and label_images
    labels, label_images = show_masks_on_image(random_image, image_mask, save_path="crossfar_mask_2.png")

    # Generate ASCII art from the image
    combined_labels = np.maximum.reduce(labels)
    ascii_art, map2d = image_to_ascii(combined_labels, label_images[0], new_w//5, new_h//5)
    # print(ascii_art, map2d)

    floor_plan_description = """
    The floor plan of the office that is the racetrack starts with a straight narrow hallway with an open space on the right side that we cannot go into, so there we must follow the left wall roughly. Then it goes into a large open space, for that space its best to go straight into nearly the wall and then turn right. When you turn right there will be more open space, just make sure you stay left. From then on its a small corridor, just don't hit things, and go slow.
    """



    # Define the system prompt
    system_prompt = """
You are a self-driving model embodied as a 5ft tall, two-wheeled robot. 
You are currently on a race track and your task is to navigate through it.
Your location is denoted by CURRENT LOCATION.
The track is represented as a grid of labelled "pixels".
You MUST stick to the left side of the pathway, which is the "Floor" pixels. 
The track is designed in a clockwise direction, so you'll likely turn right at corners. 
However, always bias towards staying near the left side using W-A of the floor.
The "pixel array" is from a fisheye camera, it is NOT a step by step representation. 
By driving forward one command, you will move a significant distance in real life.
You are given the history of the view you were given, and your command, using this you can see roughly how you navigate through the environment.

First, analyze what is in front of you. Is there anything blocking you right below you? 
Which direction are the floor pixels spanning the farthest in? 
Which command should I give to go in that direction? 
It takes many steps to move to floor pixels that are far away so don't worry about hitting walls far after the floor pixels.

Lastly, you should plan a few steps ahead. 
To get to that floor pixel you want, what immediate move should you make (or backup if you are blocked)? 
And what next few can you also make?

IFTHERE IS A WALL VERY CLOSE TO YOUR LEFT OR RIGHT SIDE, YOU WILL GET STUCK. IF YOU GET STUCK, YOU WILL LOSE POINTS. DONT DIE. MAKE SURE TO STAY AWAY FROM WALLS IMMEDIATELY!!

Return the command W A D W-A W-D or S"""

    surr = describe_surroundings(map2d)
    surr2 = generate_detailed_map_report_v2(map2d)

    # Generate a response from the AI
    import time
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": surr},
            {"role": "user", "content": ascii_art + "\n\nThe above is the current view from the camera. We are on a race track and we need to follow the left side, racing clockwise. There may be large open spaces that can distract us, but we should stick to the left side of the walls. The best action to take is to be near the left wall and go straight."},
            {"role": "user", "content": floor_plan_description},
            # {"role": "user", "content": ' '.join([f"\n\nPrevious command: {command}\nPrevious view:\n{view}" for view, command in history[-5:]])},
            {"role": "user", "content": """Above is a pixel plot of your environment with a frontward facing camera. This is a conversion of an image from a fisheye camera into an ASCII representation of the scene ahead. The most important thing is to not hit walls or get stuck on things. Be cautious! Make micro adjustments if needed. Only use what is immediately ahead of you.
If there is a wall directly in front, consider turning or backing up.
Output a python list of commands as a planned path using; W, A, D, W-A, W-D, or S to say which direction to go in. Remember W is Forward/North, A is Left/West, S is Back/South, D is Right/East. It's like a car game on a computer. Just fill this out. Put as many commands in the list as you feel confident, but at least 1 and at most 4. Tell me exactly what you're thinking, make a plan, go over it with yourself, edit it, then return the commands like i say below;


Make sure your response contains this at the end!

COMMAND: [_, _, _, ...]"""},

        ],
        temperature=0.5,
        n=1,
    )
    
    print("Time taken for GPT call: %s seconds" % (time.time() - start_time))

    print(response["choices"][0]["message"]["content"].split("COMMAND:")[0].strip())
    
    commands = [eval(choice["message"]["content"].split("COMMAND:")[1].strip()) for choice in response["choices"]]
    command = [max(set(item), key=item.count) for item in zip(*commands)]

    print(command)

    history.append((ascii_art, command, surr))

    with open('./log/history.txt', 'w') as f:
        for entry in history:
            f.write(f"{entry[0]}\nCommand: {entry[1]}\nDescription: {entry[2]}\n")

    while sum(len(entry[0]) + len(entry[1]) for entry in history) > 12000:
        history.pop(0)

    return command


from flask import Flask, jsonify, request, Response
from flask_cors import CORS

import numpy as np
import json
import base64
from io import BytesIO


app = Flask(__name__)
CORS(app, supports_credentials=True)


# @app.after_request
# def after_request_func(response):
#     play(audio)
#     return response

@app.route('/process_image', methods=['POST'])
def process_image():
    img_base64 = request.json['image']
    img_data = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_data, np.uint8)
    print(img_array.shape)
    img_array = img_array.reshape((1208, 1928, 3))

    try:
        f = modal.Function.lookup("chatPID", "ImageProcessor.process_mask")
        img_mask = f.remote(img_array)

        # Call your function
        # response = process_image_and_generate_direction(img_array)
        response_dir = process_image_and_generate_direction(img_array, img_mask, show_masks_on_image)
        

    except Exception as e:
        print(e)
        response_dir = "S"

    # Return the response
    return jsonify(response_dir)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)