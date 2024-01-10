# This is the main python file that generates the emails
import openai
from bs4 import BeautifulSoup
import requests

# Make sure you set up your API key
openai.api_key = 'YOUR_OPENAI_API_KEY'
class EmailMarketingAssistant:

    SAMPLE_EMAILS = {
    'e-commerce': {
        'convince_to_buy': [
            "Introducing our new {product_name}: {product_description}. Grab yours now!",
            "Experience the best with our new {product_name}. {product_description}. Limited stock!",
            "Why wait? The {product_name} you've always wanted is here. {product_description}.",
            "{product_name}: Where quality meets desire. {product_description}. Don't miss out!",
            "Discover the new dimension of quality with {product_name}. {product_description}. Available now!"
        ]
    },
    'people': {
        'welcome_new_user': [
            "Welcome {user_name}! We're thrilled to have you on board.",
            "Hi {user_name}, thanks for choosing us! Let's embark on this journey together.",
            "A warm welcome to our community, {user_name}!",
            "{user_name}, you've made a fantastic decision. Welcome to the family!",
            "It's a pleasure to see you, {user_name}. Welcome and let's get started!"
        ],
        'congratulate_on_purchase':[
            "Congratulations on your new {product_name} purchase, {user_name}! We're sure you'll love it.",
            "Hey {user_name}, great choice! Your new {product_name} is on its way. Enjoy!",
            "Thank you for choosing {product_name}, {user_name}! We're excited for you to try it out.",
            "{user_name}, your impeccable taste shines with your {product_name} purchase! Cheers!",
            "Rock on, {user_name}! Your {product_name} will surely turn heads!"
        ],
    },
    'blog': {
        'new_blog': [
            "Just out: our new blog post, {post_title}, covering everything about {topic}. Dive in!",
            "Unveiling our latest piece: {post_title}. Discover more about {topic}.",
            "{post_title} - a fresh take on {topic}. Read now!",
            "Explore the depths of {topic} in our new article: {post_title}. Check it out!",
            "Hot off the press: {post_title}. Delve into the world of {topic} now!"
        ]
    }
}


    def get_sample_email(self, business_type, campaign_goal, **details):
        sample_emails = self.SAMPLE_EMAILS.get(business_type, {}).get(campaign_goal, [])
        if not sample_emails:
            return ["Sorry, no sample email found for your criteria."] * 5
        
        refined_emails = []
        for sample in sample_emails:
            refined_emails.append(self.refine_prompt(sample.format(**details)))

        return refined_emails

    def refine_prompt(self, prompt):
        gpt3_message = {
            "messages": [{
                "role": "user",
                "content": f"Given this sample email: '{prompt}', create a similar yet unique marketing email."
            }]
        }
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=gpt3_message['messages']
        )
        return response.choices[0].message['content'].strip()

    def get_company_description(self, website_url):
        try:
            response = requests.get(website_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            description = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
            if description:
                return description.get('content')
            else:
                return "Description not found. Please provide manually."
        except Exception as e:
            return f"Error fetching description: {e}"

if __name__ == "__main__":
    assistant = EmailMarketingAssistant()

    mail_type = input("Enter the kind of mail to send (e-commerce, people, blog, etc.): ")
    campaign_goal = input("Enter your campaign goal (convince_to_buy, congratulate_on_purchase, welcome_new_user, new_blog): ")

    details = {}
    
    # For e-commerce related prompts
    if mail_type == "e-commerce":
        details['product_name'] = input("Enter the product name: ")
        if campaign_goal in ['convince_to_buy']:
            details['product_description'] = input("Provide a brief description of the product: ")
    
    # For new customer related prompts
    if mail_type == "people" and campaign_goal in ['welcome_new_user']:
        details['user_name'] = input("Provide new users name: ")
    elif mail_type == "people" and campaign_goal in ['congratulate_on_purchase']:
        details['user_name'] = input("Provide new users name: ")

    # For blog related prompts
    elif mail_type == "blog" and campaign_goal == "new_blog":
        details['post_title'] = input("Enter the blog post title: ")
        details['topic'] = input("Enter the post topic: ")

    # Fetch company website details
    website_url = input("Enter your company website URL (or press Enter to skip): ")
    if website_url:
        company_description = assistant.get_company_description(website_url)
        print(f"Fetched company description: {company_description}")

    email_contents = assistant.get_sample_email(mail_type, campaign_goal, **details)
    print("\nRecommended Email Contents:\n")
    for i, content in enumerate(email_contents, 1):
        print(f"Email {i}:\n{content}\n")
