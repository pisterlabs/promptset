from datetime import datetime
from story_generator import generate_story_with_title


# Function to add a new blog post to the HTML file
def add_blog_post(html_content, title, story):
    post_template = f"""
    <article class="blog-post">
        <h2>{title}</h2>
        <p class="date">Published on {datetime.now().strftime('%B %d, %Y')}</p>
        <div class="story-content">
            {story}
        </div>
    </article>
    """

    # Find the position to insert the new post within the <section> tag
    insertion_point = html_content.find("<!-- Repeat the structure for additional blog posts -->")

    # Insert the new post within the <section> tag
    updated_content = html_content[:insertion_point] + post_template + html_content[insertion_point:]

    return updated_content

# Load the existing HTML content
with open("blog.html", "r") as file:
    html_content = file.read()

# Generate a new story with title
generated_title, generated_story = generate_story_with_title()

# Add the new blog post to the HTML content
updated_html = add_blog_post(html_content, generated_title, generated_story)

# Save the updated HTML content to the same file
with open("blog.html", "w") as file:
    file.write(updated_html)

print("Blog updated successfully!")

# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# from datetime import datetime
# import json

# # Load environment variables from the .env file
# load_dotenv()

# # Access the environment variable
# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)

# # Function to generate a short story and title
# def generate_story_with_title():
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": "You are a German story writer. You write children's stories. The story should be 100 words in JSON format"},
#             {"role": "user", "content": "Generate me a short story and title. The story should be in Deutsch"},
#         ]
#     )

#     # Extracting story and title from the generated content
#     response_content = completion.choices[0].message.content

#     # Parse the JSON response
#     response_json = json.loads(response_content)

#     # Extract the title
#     title = response_json["title"]
#     story = response_json['story']

#     return title, story

# # Function to add a new blog post to the HTML file within a specific section
# def add_blog_post(html_content, title, story):
#     post_template = f"""
#     <article class="blog-post">
#         <h2>{title}</h2>
#         <p class="date">Published on {datetime.now().strftime('%B %d, %Y')}</p>
#         <div class="story-content">
#             {story}
#         </div>
#     </article>
#     """

#     # Find the position to insert the new post within the <section> tag
#     section_start = html_content.find("<section")
#     section_end = html_content.find("</section>") + len("</section>")

#     # Insert the new post at the beginning of the <section> tag
#     updated_content = (
#         html_content[:section_end].rstrip() +
#         post_template +
#         html_content[section_end:]
#     )

#     return updated_content

# # Load the existing HTML content
# with open("blog.html", "r") as file:
#     html_content = file.read()

# # Generate a new story with title
# generated_title, generated_story = generate_story_with_title()

# # Add the new blog post to the HTML content within the specified section
# updated_html = add_blog_post(html_content, generated_title, generated_story)

# # Save the updated HTML content to the same file
# with open("blog.html", "w") as file:
#     file.write(updated_html)

# print("Blog updated successfully!")


