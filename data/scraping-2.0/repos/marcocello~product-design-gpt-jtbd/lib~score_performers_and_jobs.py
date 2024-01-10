import os
import json 
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL")
TEMPERATURE = 0

model = ChatOpenAI(model_name=MODEL, temperature=TEMPERATURE)

prompt = ChatPromptTemplate.from_messages(
[
    ("system", """


Act like an accomplished business expert, visionary, and a seasoned advisor well-versed, like Bill Aulet the author of Disciplined Enterpreneurs.
     
Your role is to facilitate the generation of Jobs and Job Performers by incorporating these characteristics. 
     
     """),

    ("human", """
     Based on input from users, you will generate different combinations of Job Performers and Jobs, focusing on creating optimal matches for effective outcomes.

     Based on the following vision:
     ============
     {vision}
     ============
    
     And based on the following tuples of Job_Performers and Jobs:
     ============
     {performers_and_jobs}
     ============
    
     For each Job_Performer and related Jobs answer the following questions:

     Funding Viability: Is the target customer well-funded? If the customer does not have money, the market is not attractive because it will not be sustainable and provide positive cash flow for the new venture to grow.

     Sales Force Accessibility: Is the target customer readily accessible to your sales force? You want to deal directly with customers when starting out, rather than rely on third parties to market and sell your product, because your product will go through iterations of improvement very rapidly, and direct customer feedback is an essential part of that process. Also, since your product is substantially new and never seen before (and potentially disruptive), third parties may not know how to be effective at creating demand for your product.
     
     Customer Value Proposition: Does the target customer have a compelling reason to buy? Would the customer buy your product instead of another similar solution? Or, is the customer content with whatever solution is already being used? Remember that on many occasions, your primary competition will be the customer doing nothing.

     Whole Product Delivery: Can you today, with the help of partners, deliver a whole product? The example here that I often use in class is that no one wants to buy a new alternator and install it in their car, even if the alternator is much better than what they currently have. They want to buy a car. That is, they want to buy a whole functional solution, not assemble one themselves. You will likely need to work with other vendors to deliver a solution that incorporates your product, which means that you will need to convince other manufacturers and distributors that your product is worth integrating into their workflows.
     
     Competitive Landscape: Is there entrenched competition that could block you? Rare is the case where no other competitors are vying to convince a customer to spend their budget on some product to meet the identified need. How strong are those competitors, from the customer's viewpoint (not your viewpoint or from a technical standpoint)? Can the competition block you from starting a business relationship with a customer? And how do you stand out from what your customer perceives as alternatives?

     Market Segmentation and Scalability: If you win this segment, can you leverage it to enter additional segments? If you dominate this market opportunity, are there adjacent opportunities where you can sell your product with only slight modifications to your product or your sales strategy? Or will you have to radically revise your product or sales strategy in order to take advantage of additional market opportunities? While you want to stay focused on your beachhead market, you do not want to choose a starting market from which you will have a hard time scaling your business. Geoffrey Moore uses the metaphor of a bowling alley, where the beachhead market is the lead pin, and dominating the beachhead market knocks down the lead pin, which crashes into other pins that represent either adjacent market opportunities or different applications to sell to the customer in your beachhead market.

    For each Job_Performer rate how the Job Performer might behave for each question. Use a scale from 1 to 5, where 1 is the least favorable and 5 is the most favorable.
     
    The output should be:
     [
     {{
     "Job_Perfomers": "",
     "Funding Viability": "",
     "Funding Viability Score": "",
     "Sales Force Accessibility": "",
     "Sales Force Accessibility Score": "",
     "Customer Value Proposition": "",
     "Customer Value Proposition Score": "",
     "Whole Product Delivery": "",
     "Whole Product Delivery Score": "",
     "Competitive Landscape": "",
     "Competitive Landscape Score": "",
     "Market Segmentation and Scalability",
     "Market Segmentation and Scalability Score"
     }}
     ]

    You have to analyze all the Job_performers
     
    {additional_prompt}     
     """)
])

functions = [
    {
    "name": "score",
    "description": "Score job performers and main jobs",
    "parameters": {
        "type": "object",
        "properties": {
            "score_performers_and_jobs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "Job_Perfomers": {
                            "type": "string"
                        },
                        "Funding Viability": {
                            "type": "string"
                        },
                        "Funding Viability Score": {
                            "type": "number"
                        },
                        "Sales Force Accessibility": {
                            "type": "string"
                        },
                        "Sales Force Accessibility Score": {
                            "type": "number"
                        },
                        "Customer Value Proposition": {
                            "type": "string"
                        },
                        "Customer Value Proposition Score": {
                            "type": "number"
                        },
                        "Whole Product Delivery": {
                            "type": "string"
                        },
                        "Whole Product Delivery Score": {
                            "type": "number"
                        },
                        "Competitive Landscape": {
                            "type": "string"
                        },
                        "Competitive Landscape Score": {
                            "type": "number"
                        },
                        "Market Segmentation and Scalability": {
                            "type": "string"
                        },
                        "Market Segmentation and Scalability Score": {
                            "type": "number"
                        },
                    }
                }
            },
        },
    },
    },
]

chain = (
    prompt |
    model.bind(function_call={"name": "score"}, functions = functions)
)