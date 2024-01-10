from langchain import PromptTemplate

template1 = """"Using the company name, {CompanyName}, and relevant details provided below, 
please research the market and analyze previous campaigns to develop a comprehensive and innovative marketing campaign strategy. 
Consider the company's target audience, unique selling points, and industry trends in order to make informed suggestions for maximizing impact and achieving marketing goals.
Provide a detailed campaign plan including objectives, target channels, content ideas, and a timeline for implementation."
To achieve this, these are the tools that are at your disposal:
- Search Engine: SerpAPI


Company Details:

Company Name: {CompanyName}
Industry: {Industry}
Target Audience: {TargetAudience}
Unique Selling Points:{UniqueSellingPoints}
Previous Campaigns: {PreviousCampaigns}
Marketing Goals: {MarketingGoals}"""


initial_template = PromptTemplate(
    input_variables =["CompanyName", "Industry", "TargetAudience", "UniqueSellingPoints", "PreviousCampaigns", "MarketingGoals"],
    template = template1,
    )

#print(initial_template.format(CompanyName ="Apple", Industry= "Technology", TargetAudience="Tech Enthusiasts",UniqueSellingPoints="High Quality Products", PreviousCampaigns= "iPhone 12", MarketingGoals= "Increase sales by 10%"))



template2 = """Based on the initial research and analysis conducted on {CompanyName} and its previous campaigns, please refine the marketing campaign ideas by focusing on the most effective strategies and tactics.
Identify the key elements that will contribute to a successful campaign and provide specific recommendations for improvement. 
Outline a revised campaign plan that incorporates these refinements, including updated objectives, target channels, content ideas, and a timeline for implementation.
"""

marketing_refinement_template = PromptTemplate(
    input_variables=["CompanyName"],
    template=template2,
    )

template3 = """Considering the previous content and marketing strategies used by {CompanyName}, 
analyze their performance and identify key factors that contributed to their success or shortcomings.
Based on this analysis, brainstorm and suggest the ideal post that will resonate with the target audience, align with the company's unique selling points, 
and support the current marketing campaign objectives. Provide a detailed description of the post concept, including format, message, visuals, and any calls-to-action,
ensuring that it stands out among the company's previous content and contributes to the overall success of the campaign.
"""

post_idea_template = PromptTemplate(
    input_variables=["CompanyName"],
    template=template3,
    )

template4 = """Please conduct web research to gather information on the company, {CompanyName}
and identify its main competitors within the {Industry} sector. Provide their names and a brief description of their products, services, pros and cons.
Now that you have a better understanding of the company's competitors, research their marketing strategies and identify the most effective ones. Also identify any strategies that were not successful.
Based on that knowledge and your previous research on {CompanyName}, brainstorm and suggest a new marketing strategy that will help the company stand out from its competitors.
Provide a comprehensive overview of the company's competitive landscape, highlighting any trends or commonalities among their marketing approaches. 
When all the information is gathered, provide a detailed description of the new marketing strategy, including objectives, target channels, content ideas, and a timeline for implementation.
"""

get_initial_info_template = PromptTemplate(
    input_variables=["CompanyName", "Industry"],
    template=template4,
)
