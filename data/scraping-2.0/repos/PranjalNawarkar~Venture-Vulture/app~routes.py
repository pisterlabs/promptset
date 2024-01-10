import tweepy
import openai
import random
from flask import Blueprint, render_template, request, redirect, url_for, current_app

main = Blueprint('main', __name__)

def get_startup_tweets():
    bearer_token = current_app.config['TWITTER_BEARER_TOKEN']
    client = tweepy.Client(bearer_token=bearer_token)
    query = 'from:acquiredotcom "🔥 New Startup Listed 🔥" -is:retweet'
    tweets = client.search_recent_tweets(query=query, tweet_fields=['context_annotations', 'created_at'], max_results=10)
    return [tweet.text for tweet in tweets.data] if tweets.data else []

@main.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {
            'experience': request.form.get('experience'),
            'industry_expertise': request.form.get('industry_expertise'),
            'educational_background': request.form.get('educational_background'),
            'employment_status': request.form.get('employment_status'),
            'financial_resources': request.form.get('financial_resources'),
            'risk_tolerance': request.form.get('risk_tolerance'),
            'preferred_industries': request.form.get('preferred_industries'),
            'social_impact': request.form.get('social_impact'),
            'innovation_interests': request.form.get('innovation_interests'),
            'business_model': request.form.get('business_model'),
            'growth_ambition': request.form.get('growth_ambition'),
            'time_commitment': request.form.get('time_commitment')
        }
        return redirect(url_for('main.results', **data))
    return render_template('index.html')

@main.route('/results')
def results():
    data = request.args
    startup_tweets = get_startup_tweets()
    idea = generate_idea(data, startup_tweets)
    return render_template('results.html', idea=idea)

def generate_idea(data, startup_tweets):
    openai.api_key = current_app.config['OPENAI_API_KEY']
    tweet_details = [tweet for tweet in startup_tweets]

    random.shuffle(tweet_details)
    selected_tweet_inspiration = '; '.join(tweet_details[:5])  # Using a subset of 5 tweets for inspiration

    prompt = construct_prompt(data, selected_tweet_inspiration)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "You are Venture Vulture, an AI designed to synthesize unique, novel, and viable business ideas."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.6
        )
        return response.choices[0].message['content'] if 'choices' in response else "Error: No idea generated."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error generating idea."

def construct_prompt(data, tweet_inspiration):
    return (
        f"Venture Vulture, create a diverse and innovative business idea based on various startup trends "
        f"from Acquiredotcom. User Background: {data.get('experience', 'No Experience')} in {data.get('industry_expertise', 'No Industry')}. "
        f"Focus on {data.get('preferred_industries', 'No Industries')}. Draw inspiration from these varied startups: {tweet_inspiration}. "
        f"Aim for a unique, novel, and practical idea."
    )

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
