from canvasapi import Canvas
from decouple import config
import openai
import time

API_URL = config('API_URL')
API_KEY = config('API_KEY')
openai.api_key = config('openai.api_key')

canvas = Canvas(API_URL, API_KEY)

course_id = config('COURSE_ID')
course = canvas.get_course(course_id)

if course is None:
    print("❌ Error connecting to Canvas")  
else:
    print("✅ Connected to Canvas")

def remove_tags(text):
    if text is None:
        return None
    text = text.replace("<p>", "").replace("</p>", "\n").replace("BOT: ", "")
    if "<p>" in text or "</p>" in text or "BOT: " in text:
        print("❌ Error removing tags")
    else:
        print("✅ Tags removed")
    return text

def write_to_file(text):
    if text is None:
        return None
    try:
        with open("post_message.txt", "w") as f:
            f.write(text)
        print("✅ File written")
    except Exception as e:
        print(f"❌ Error writing file: {e}")

def generate_response(text):
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=text,
            temperature=0.7,
            max_tokens=150,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            stop=["\n", " Student: ", " BOT:"]
        )
        return response.choices[0].text
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def create_discussion():
    print("What is the name of the discussion?")
    title = input()
    print("What is the message of the discussion?")
    message = input()
    discussion = course.create_discussion_topic(
        title = title,
        message = message,
        discussion_type = "side_comment"
    )
    # check if it worked
    if discussion is None:
        print("❌ Error creating discussion")
    else:
        print("✅ Discussion created")
    return discussion

num_discussion_posts = list(course.get_discussion_topics())
print(f"Total number of discussion posts: {len(num_discussion_posts)}")

keyword = "BOT"
# See if any discussion posts contain the keyword
discussion_posts_list = []
for post in num_discussion_posts:
    if keyword in post.message:
        print("✅ Found bot post")
        discussion_posts_list.append(post)

print(f"Number of discussion posts with keyword: {len(discussion_posts_list)}")

print("✅ Check 1")

print("Would you like to create a new discussion? (y/n)")
answer = input()
if answer == "y":
    discussion = create_discussion()
    discussion_posts_list.append(discussion)
    print("✅ Discussion added to list")

print("✅ Check 2")

# Check if discussion_posts is not None and it has items
if discussion_posts_list is not None and len(discussion_posts_list) > 0:
    print("Would you like to delete the discussion post? (y/n)")
    answer = input()
    if answer == "y":
        discussion_posts_list[0].delete()
        print("✅ Discussion post deleted")

print("✅ Check 3")

# Loop through discussion posts that contain the keyword, if so, write it to file
for post in discussion_posts_list:
    if keyword in post.message:
        print("✅ Found bot post")
        prompt = remove_tags(post.message)
        write_to_file(prompt)
        # reply to the discussion post
        response = generate_response(prompt)
        print(response)
        # print data type of prompt
        print(type(post))
        #try catch block to reply to the discussion post
        try:
            post.create_discussion_entry(message = response)
            print("✅ Discussion post replied to")
        except Exception as e:
            print(f"❌ Error replying to discussion post: {e}")
    else:
        print("❌ No bot post found")
print("💤 All done...")

# TODO: Fix the issue with the discussion post not being replied to