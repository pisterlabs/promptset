from langchain.prompts import PromptTemplate

CONCISE_TEMPLATE= """I want you to act as a SMS service analyst, you classify the messages into categories.
The classification task is a multi-label classification task(primary_category and secondary_category)
The primary_category focus on the functionality of the message, and the secondary_category focus on the content type(industry, business type, app type) of the message.

The Primary category can be one of the following classes:
[Advertisement, Notification, Verification, Subscription, Transaction, Reminder, Alert, Survey, Support, Invitation, Personal, Spam]

The Secondary category can be an array of the following classes:
[Entertainment, Banking and Finance, Retail, Telecom, Travel and Hospitality, Government and Public Services, Healthcare, Education, Social Networking, Utilities, News and Media, Non-Profit, Technology, Automotive, Food and Dining, Sports, Fashion and Beauty, Real Estate, Legal and Insurance, Job and Recruitment, Loan Service]

You should format your answer in JSON FORMAT with keys being primary_category and secondary_category
The message to be classified is delimited by triple backticks

message = ```{message}```
"""

CONCISE_PROMPT = PromptTemplate(input_variables=["message"], template=CONCISE_TEMPLATE)
