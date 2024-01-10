import openai
import random
import praw

openai.api_key = "OpenAI-API"

def generate_reddit_prompt(post_title, subreddit_name):
    prompt = f"""Generate a 900-second deform prompt for a Reddit post edit featuring the post titled '{post_title}' from the subreddit r/{subreddit_name}. Make the format look like this and make the prompt have no text.{{
        "0": "Opening slide showing the post title '{post_title}', 'Let's Dive In' text overlay",
        "50": "Zooming into the comments, 'Community Takes' text",
        "100": "Highlighting the top upvoted comment, 'Most Agreed' text",
        "150": "Showing controversial comments, 'Hot Debate' text",
        "200": "Screen split showing post images or content, 'The Core' text",
        "250": "Slide of similar posts, 'Related Topics' text",
        "300": "User reactions via emojis or upvotes, 'The Verdict' text",
        "350": "Close-up of the post author's profile, 'Who Posted?' text",
        "400": "Slide showing subreddit rules, 'Play by the Rules' text",
        "450": "Zoom out to show the subreddit banner, 'Big Picture' text",
        "500": "Quick scroll through other top posts in the subreddit, 'Community Vibes' text",
        "550": "Slide of related subreddits, 'Expand Your View' text",
        "600": "Post analytics like upvotes and comments over time, 'Post Journey' text",
        "650": "Closing remarks or summary, 'Final Thoughts' text",
        "700": "End slide with a call to action like 'Upvote or Comment', 'Your Move' text",
        "750": "Slide encouraging to share the post, 'Spread the Word' text",
        "800": "Acknowledgment of the Reddit community, 'Thanks Reddit!' text",
        "850": "Legal disclaimers or credits, 'The Fine Print' text",
        "900": "Final slide summarizing the post's impact, 'The Ripple Effect' text overlay, epic music crescendo"
    }}"""

    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()

generated_prompt = generate_reddit_prompt("I cheated on my boyfriend", "stories")
print(generated_prompt)

# Initialize the Reddit API client
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)
# Select a subreddit
subreddit = reddit.subreddit('stories')

# Fetch random posts (change the limit to fetch more posts)
random_posts = list(subreddit.hot(limit=50))

# Pick a random post from the fetched list
random_post = random.choice(random_posts)

# Extract details
post_title = random_post.title
subreddit_name = random_post.subreddit.display_name
story_content = random_post.selftext

print(f"Title: {post_title}")
print(f"Subreddit: {subreddit_name}")
print(f"Story Content: {story_content}")

generated_prompt = generate_reddit_prompt(post_title, subreddit_name)
print(generated_prompt)

