from langchain.prompts import PromptTemplate


DETAILED_TEMPLATE= """I want you to act as a SMS service analyst, you classify the messages into categories.
The task is a multi-label classification task(primary_category and secondary_category)
The primary_category focus on the functionality of the message, and the secondary_category focus on the content type(industry, business type, app type) of the message.

The Primary category can be one of the following:
Advertisement: Messages promoting products, services, or events
Notification: General information or updates from apps, services, or systems
Verification: Messages containing codes or links for authentication
Subscription: Messages related to user subscriptions or content updates
Transaction: Messages about financial transactions or account updates
Reminder: Messages reminding users about appointments or events
Alert: Urgent or critical messages requiring immediate action
Survey: Messages requesting user feedback or participation
Support: Messages related to customer support or query resolution
Invitation: Messages inviting users to events or exclusive offers
Personal: Messages for personal communication or non-commercial use
Spam: Unsolicited or unwanted messages

The Secondary category can be an array of the following classes:
Entertainment: Messages from entertainment companies, event organizers, or content providers
Banking and Finance: Messages from banks, financial institutions, or investment firms
Retail: Messages from retail stores, e-commerce platforms, or online marketplaces
Telecom: Messages from telecommunication companies or mobile service providers
Travel and Hospitality: Messages from airlines, travel agencies, hotels, or booking services
Government and Public Services: Messages from government agencies or public service providers
Healthcare: Messages from healthcare providers, clinics, or medical institutions
Education: Messages from educational institutions, schools, or online learning platforms
Social Networking: Messages from social media platforms, networking sites, or online communities
Utilities: Messages from utility service providers, such as electricity, water, or gas companies
News and Media: Messages from news outlets, media organizations, or journalists
Non-Profit: Messages from non-profit organizations or charitable institutions
Technology: Messages from technology companies, software developers, or gadget manufacturers
Automotive: Messages from automotive companies or dealerships
Food and Dining: Messages from restaurants, food delivery services, or catering businesses
Sports: Messages from sports organizations, teams, or event organizers
Fashion and Beauty: Messages from fashion brands, beauty products, or cosmetics companies
Real Estate: Messages from real estate agencies, property developers, or brokers
Legal and Insurance: Messages from law firms, insurance companies, or legal services
Job and Recruitment: Messages related to job offers, career opportunities, or recruitment agencies
Loan Service: Messages from loan service providers, banks offering loan services or related.

You should format your answer in JSON FORMAT with keys being primary_category and secondary_category
The message to be classified is delimited by triple backticks

message = ```{message}```"""

DETAILED_PROMPT = PromptTemplate(input_variables=["message"], template=DETAILED_TEMPLATE)