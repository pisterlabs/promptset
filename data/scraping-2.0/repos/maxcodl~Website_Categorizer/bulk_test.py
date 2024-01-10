import openai
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import time
import pandas as pd
import re

# Set your OpenAI API key
openai.api_key = 'OPENAI_API_KEY'
model_engine = "gpt-3.5-turbo-16k"

categories_keywords = {
            "Jobs & Education": ["Education", "Colleges & Universities", "Distance Learning","Homeschooling", "Primary & Secondary Schooling (K-12)","Standardized & Admissions Tests", "Teaching & Classroom Resources","Training & Certification","Vocational & Continuing Education","Jobs", "Career Resources & Planning","Job Listings","Resumes & Portfolios"],
            "Law & Government": ["Government","Courts & Judiciary","Visa & Immigration","Legal","Bankruptcy","Legal Education","Legal Services","Military","Public Safety","Crime & Justice", "Emergency Services",  "Law Enforcement", "Security Products & Services",  "Social Services"],
            "Arts & Entertainment": [ "Anime & Manga","Acting & Theater","Architecture","Art Museums & Galleries","Bars, Clubs & Nightlife","Cartoons","CD & Audio Shopping","Circus","Classical Music","Comics","Concerts & Music Festivals","Country Music","Dance","Dance & Electronic Music","Design","Experimental & Industrial Music","Expos & Conventions","Film & TV Industry","Film Festivals","Flash-Based Entertainment","Fun Tests & Silly Surveys","Funny Pictures & Videos","Jazz & Blues","Magic","Movie Listings & Theater Showtimes","Music Education & Instruction","Music Equipment & Technology","Music Reference","Music Streams & Downloads","Music Videos","Occult & Paranormal","Online Image Galleries","Online Video","Opera","Painting","Photographic & Digital Arts","Political Humor","Pop Music","Radio","Recording Industry","Religious Music","Rock Music","Soundtracks","TV Commercials","TV Shows & Programs","Urban & Hip-Hop","World Music"],
            "Adult": ["Porn","Creampie","Lesbian","Hentai","Adult"],
            "Autos & Vehicles": ["Bicycles & Accessories","Bike Parts & Repair","BMX Bikes","Boats & Watercraft","Campers & RVs","Cargo Trucks & Trailers","Classic Vehicles","Commercial Vehicles","Gas Prices & Vehicle Fueling","Hybrid & Alternative Vehicles","Motor Vehicles (By Type)","Motorcycles","Off-Road Vehicles","Trucks & SUVs","Used Vehicles","Vehicle Codes & Driving Laws","Vehicle Licensing & Registration","Vehicle Parts & Accessories","Vehicle Parts & Services","Vehicle Repair & Maintenance","Vehicle Shopping","Vehicle Shows"],
            "Beauty & Fitness": ["Beauty Pageants","Body Art","Cosmetic Procedures","Cosmetology & Beauty Professionals","Face & Body Care","Fashion & Style","Fitness","Hair Care","Spas & Beauty Services","Weight Loss","Cosmetic Surgery","Hygiene & Toiletries","Make-Up & Cosmetics","Perfumes & Fragrances","Skin & Nail Care","Unwanted Body & Facial Hair Removal","Fashion Designers & Collections","Hair Loss","Massage Therapy"],
            "Business & Industrial": ["Advertising & Marketing","Aerospace & Defense","Agriculture & Forestry","Automotive Industry","Business Education","Business Finance","Business Operations","Business Services","Chemicals Industry","Construction & Maintenance","Energy & Utilities","Hospitality Industry","Industrial Materials & Equipment","Manufacturing","Metals & Mining","Pharmaceuticals & Biotech","Printing & Publishing","Retail Trade","Small Business","Textiles & Nonwovens","Transportation & Logistics","Public Relations","Space Technology","Agricultural Equipment","Forestry","Livestock","Venture Capital","Business Plans & Presentations","Management","Consulting","Corporate Events","E-Commerce Services","Fire & Security Services","Office Services","Office Supplies","Writing & Editing Services","Cleaning Agents","Plastics & Polymers","Building Materials & Supplies","Electricity","Oil & Gas","Renewable & Alternative Energy","Event Planning","Food Service","Heavy Machinery","Precious Metals","Retail Equipment & Technology","MLM & Business Opportunities","Freight & Trucking","Mail & Package Delivery","Maritime Transport","Moving & Relocation","Packaging","Parking","Rail Transport","Urban Transport"],
            "Computers & Electronics": ["CAD & CAM","Computer Hardware","Computer Security","Consumer Electronics","Electronics & Electrical","Enterprise Technology","Networking","Programming","Software","Computer Components","Computer Drives & Storage","Computer Peripherals","Desktop Computers","Laptops & Notebooks","Hacking & Cracking","Audio Equipment","Camera & Photo Equipment","Car Electronics","Drones & RC Aircraft","Game Systems & Consoles","GPS & Navigation","TV & Video Equipment","Electronic Components","Power Supplies","Data Management","Data Formats & Protocols","Network Monitoring & Management","VPN & Remote Access","Java","Business & Productivity Software","Device Drivers","Internet Software","Multimedia Software","Operating Systems","Software Utilities"],
            "Finance": ["Accounting & Auditing","Banking","Credit & Lending","Financial Planning & Management","Grants, Scholarships & Financial Aid","Insurance","Investing","Billing & Invoicing","Tax Preparation & Planning","Credit Cards","Credit Reporting & Monitoring","Loans","Retirement & Pension","Study Grants & Scholarships","Health Insurance","Commodities & Futures Trading","Currencies & Foreign Exchange","Stocks & Bonds","M&A","M&A Advisory","Raise Funds"],
            "Food & Drink": ["Beverages","Cooking & Recipes","Food","Food & Grocery Retailers","Restaurants","Alcoholic Beverages","Coffee & Tea","Juice","Soft Drinks","BBQ & Grilling","Desserts","Soups & Stews","Baked Goods","Breakfast Foods","Candy & Sweets","Grains & Pasta","Meat & Seafood","Snack Foods","Fast Food","Pizzerias","Restaurant Reviews & Reservations"],
            "Games": ["Arcade & Coin-Op Games","Board Games","Card Games","Computer & Video Games","Family-Oriented Games & Activities","Gambling","Online Games","Puzzles & Brainteasers","Roleplaying Games","Table Games","Word Games","Chess & Abstract Strategy Games","Miniatures & Wargaming","Collectible Card Games","Poker & Casino Games","Casual Games","Driving & Racing Games","Fighting Games","Music & Dance Games","Sandbox Games","Shooter Games","Simulation Games","Sports Games","Strategy Games","Video Game Emulation","Drawing & Coloring","Dress-Up & Fashion Games","Lottery","Massively Multiplayer Games","Billiards"],
            "Health": ["Aging & Geriatrics","Health Conditions","Health Education & Medical Training","Health Foundations & Medical Research","Medical Devices & Equipment","Medical Facilities & Services","Men's Health","Mental Health","Nursing","Nutrition","Oral & Dental Care","Pharmacy","Public Health","Reproductive Health","Substance Abuse","Vision Care","Women's Health","AIDS & HIV","Allergies","Arthritis","Cancer","Diabetes","Ear Nose & Throat","Eating Disorders","Endocrine Conditions","Genetic Disorders","Heart & Hypertension","Infectious Diseases","Neurological Conditions","Obesity","Pain Management","Respiratory Conditions","Skin Conditions","Sleep Disorders","Doctors' Offices","Hospitals & Treatment Centers","Medical Procedures","Physical Therapy","Anxiety & Stress","Depression","Assisted Living & Long Term Care","Special & Restricted Diets","Vitamins & Supplements","Drugs & Medications","Occupational Health & Safety","Drug & Alcohol Testing","Drug & Alcohol Treatment","Smoking & Smoking Cessation","Steroids & Performance-Enhancing Drugs","Eyeglasses & Contacts"],
            "Hobbies & Leisure": ["Clubs & Organizations","Crafts","Merit Prizes & Contests","Outdoors","Paintball","Radio Control & Modeling","Special Occasions","Water Activities","Youth Organizations & Resources","Fiber & Textile Arts","Fishing","Hiking & Camping","Model Trains & Railroads","Holidays & Seasonal Events","Weddings","Boating","Surf & Swim"],
            "Home & Garden": ["Bed & Bath","Domestic Services","Gardening & Landscaping","Home & Interior Decor","Home Appliances","Home Furnishings","Home Improvement","Home Safety & Security","Home Storage & Shelving","Home Swimming Pools, Saunas & Spas","HVAC & Climate Control","Kitchen & Dining","Laundry","Nursery & Playroom","Pest Control","Yard & Patio","Bathroom","Cleaning Services","Curtains & Window Treatments","Kitchen & Dining Furniture","Lamps & Lighting","Living Room Furniture","Rugs & Carpets","Construction & Power Tools","Doors & Windows","Flooring","House Painting & Finishing","Plumbing","Fireplaces & Stoves","Cookware & Diningware","Major Kitchen Appliances","Small Kitchen Appliances","Washers & Dryers","Lawn Mowers"],
            "Internet & Telecom": ["Communications Equipment","Email & Messaging","Mobile & Wireless","Service Providers","Web Services","Radio Equipment","Text & Instant Messaging","Voice & Video Chat","Mobile & Wireless Accessories","Mobile Apps & Add-Ons","Mobile Phones","Cable & Satellite Providers","Domain Parking","Affiliate Programs","Web Design & Development"],
            "Jobs & Education": ["Education","Jobs","Jobs & Education","Colleges & Universities","Distance Learning","Homeschooling","Primary & Secondary Schooling (K-12)","Standardized & Admissions Tests","Teaching & Classroom Resources","Training & Certification","Vocational & Continuing Education","Career Resources & Planning","Job Listings","Resumes & Portfolios"],
            "News": ["Business News","Gossip & Tabloid News","Health News","Politics","Sports News","Weather","Company News","Financial Markets News","Scandals & Investigations"],
            "Online Communities": ["Blogging Resources & Services","Dating & Personals","File Sharing & Hosting","Online Goodies","Photo & Video Sharing","Social Networks","Virtual Worlds","Matrimonial Services","Personals","Photo Rating Sites","Clip Art & Animated GIFs","Skins, Themes & Wallpapers","Social Network Apps & Add-Ons","Photo & Image Sharing"],
            "People & Society": ["Family & Relationships","Kids & Teens","Religion & Belief","Seniors & Retirement","Social Issues & Advocacy","Social Sciences","Subcultures & Niche Interests","Family","Marriage","Troubled Relationships","Children's Interests","Teen Interests","Charity & Philanthropy","Discrimination & Identity Relations","Green Living & Environmental Issues","Human Rights & Liberties","Poverty & Hunger","Work & Labor Issues","Economics","Political Science","Psychology"],
            "Articles": ["1 min read","2 min read","3 min read","4 min read","5 min read","6 min read","7 min read","8 min read","9 min read","10 min read","Medium app"],
            "Professional Networking": ["Linkedin","open to work","people to see","job opportunities","Connect with people"],
}
# Create an empty list to store data
data = []


# List of websites to extract data from
websites = [ 
'websites'
]


# Create a pandas DataFrame to store data
columns = ["URL", "Request", "Response", "Extracted Data"]
df = pd.DataFrame(columns=columns)

def categorize_based_on_keywords(text, categories_keywords):
    matched_categories = []
    for category, keywords in categories_keywords.items():
        matched_keywords = []
        for keyword in keywords:
            keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(keyword_pattern, text.lower()):
                matched_keywords.append(keyword)
        if len(matched_keywords) >= 3:
            print(f"For category '{category}', found matching keywords: {matched_keywords}")
            matched_categories.append(category)
        if not matched_categories:
            print(f"No keywords found for '{category}'.")
    return matched_categories

# Function to extract text from a website
def extract_text_from_website(url):
    try:
        response = requests.get(url, timeout=10)  # Set a timeout value in seconds
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            all_text = ' '.join([element.get_text() for element in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span'])])
            return all_text
        else:
            return None
    except requests.exceptions.Timeout:
        print(f"Timeout error: Could not connect to {url}")
        return None
    except Exception as e:
        print("Error:", e)
        return None



# Iterate through the list of websites
for website in websites:
    time.sleep(10)
    url = f"https://{website}"
    text_content = extract_text_from_website(url)
    if text_content is not None:
        categories = categorize_based_on_keywords(text_content, categories_keywords)
    else:
        categories = None

    if categories:
        print(f"Categories assigned based on keyword match: {categories}")
    else:
        print("No category was assigned based on keywords. Using OpenAI to categorize.")

    # If no categories were found and text content is None, handle the error
    if text_content is None:
        text_content = " "
    else:
        text_content = " ".join(text_content.split())
    # Limit the text content to 500 words
        word_limit = 600
        if len(text_content.split()) > word_limit:
            text_content = " ".join(text_content.split()[:word_limit])
        prompt = (
        f"This is a website [{url}], and this is the content I extracted from the website: [{text_content}]. "
        "Can you determine which category this website belongs to based on the content? "
        "I would like the result in JSON format as follows: "
        "'website': 'url', 'category': 'category_name'. "
        "Please choose a category from the list provided below for accurate classification: "
        "(Technology, Startup, Sales, Health, Business, Education, Finance, Web3, Human Resource, Generative AI, Others, Economy, Gen AI, HR, Law, Management, Productivity, Sales & Marketing, Stocks, Tech, VC & PE, Adult). "
        "You can select multiple categories if you are familiar with the website. "
        "You may also disregard the extracted data if necessary. "
        "IMPORTANT: While showing the result, only show the json."
    )

            
        # Use OpenAI to generate the category
        response = openai.ChatCompletion.create(
    model=model_engine,
    temperature=0.8,
    top_p=1,
    max_tokens=50,
    presence_penalty=0,
    frequency_penalty=0.57,
    messages=[
    {
      "role": "system",
      "content": "You are an expert in website categorization."
    },
    {
      "role": "user",
      "content": "This is a website [https://example.com], and this is the content I extracted from the website: [ ]. Can you determine which category this website belongs to based on the content? I would like the result in JSON format as follows: {'website': 'url', 'category': 'category_name'}. Please choose a category from the list provided below for accurate classification: (Technology, Startup, Sales, Health, Business, Education, Finance, Web3, Human Resource, Generative AI, Others, Economy, Gen AI, HR, Law, Management, Productivity, Sales & Marketing, Stocks, Tech, VC & PE, Entertainment, Adult). You can select multiple categories if you are familiar with the website. You may also disregard the extracted data if necessary. IMPORTANT: While showing the result, only show the json result."
    },
    {
      "role": "assistant",
      "content": "{\"website\": \"https://example.com\", \"category\": \"Others\"}"
    },
    {
        "role": "user", "content": prompt
    }
  ],
)
        print("Prompt to OpenAI:", prompt)
        generated_category = response['choices'][0]['message']['content']
        data.append({"URL": url, "Request": prompt, "Generated Category": generated_category, "Category": categories, "Extracted Data": text_content})
        
        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)
        
        # Save the data to an Excel file after each request
        df.to_excel("website_data_temp.xlsx", index=False)
        
        print(f"Processed {url}")
        
        # Add a delay to avoid overloading the server
        time.sleep(10)
else:
        print(f"Failed to extract text from {url}")

# Save the final data to an Excel file
df.to_excel("website_data.xlsx", index=False)
print("Final data saved to website_data.xlsx")
