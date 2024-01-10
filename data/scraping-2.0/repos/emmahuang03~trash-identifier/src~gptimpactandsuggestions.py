import openai
from keys import *
from chatgpt import categorizeObject

def giveImpactAndSuggestionsOpenai():
    openai.api_key = api_key_openai
    openai.api_base = api_base_openai
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15' # this may change in the future
    deployment_name='openai-gpt35'

    # set the role of AI assistant and give examples 
    role = {"role": "system", "content": "You are an AI assistant that helps people understand the impact of their choices when recycling, trashing items, and composting them. You also sometimes give information on what they could do to make better choices."}
    q1 = {"role": "user", "content": "give me specific stats for recycling a water bottle in 5 bullet points"}
    a1 = {"role": "assistant", "content": "1. Recycling a water bottle conserves approximately 0.15 kg (0.33 lbs) of plastic material.\n2. Recycling a single water bottle saves about 16-24 MJ of energy.\n3. Recycling a water bottle helps avoid emitting approximately 0.3-0.45 kg (0.66-1 lb) of carbon dioxide equivalent (CO2e).\n4. Recycling a water bottle reduces landfill waste by approximately 0.15 kg (0.33 lbs).\n5. Recycling a water bottle contributes to the conservation of natural resources, such as petroleum, which is used to make plastic."}
    q2 = {"role": "user", "content": "give me specific stats for composting an apple in 5 bullet points"}
    a2 = {"role": "assistant", "content": "1. Composting a single apple diverts approximately 0.15 kg (0.33 lbs) of organic waste from landfills.\n2. The decomposition of an apple through composting produces nutrient-rich compost, typically around 0.01 kg (0.02 lbs) in weight.\n3. Composting an apple helps reduce greenhouse gas emissions, as organic waste in landfills produces methane, a potent greenhouse gas.\n4. The compost generated from an apple enriches the soil with essential nutrients, such as potassium, phosphorus, and nitrogen.\n5. Incorporating compost into soil enhances its water-holding capacity, promoting better moisture retention and reducing the need for excessive irrigation."}
    q3 = {"role": "user", "content": "give me specific stats on the problem with throwing clothes in the trash in 3 bullet points and give me ways to avoid throwing clothes in the trash in 3 bullet points"}
    a3 = {"role": "assistant", "content": "Stats on the problem with throwing clothes in the trash:\n1. Textile Waste Volume: Globally, approximately 92 million tons of textile waste is generated each year, with the equivalent of one garbage truck of textiles being landfilled or burned every second. (Source: Ellen MacArthur Foundation)\n2. Low Recycling Rates: Only about 13% of the total fiber input used for clothing is recycled globally, meaning the majority of discarded clothes end up in landfills or incinerators. (Source: Pulse of the Fashion Industry Report 2020)\n3. Environmental Impact: The fashion industry accounts for about 10% of global greenhouse gas emissions, more than international flights and maritime shipping combined. Additionally, clothing production uses an estimated 1.5 trillion liters of water annually. (Source: World Resources Institute)\nWays to avoid throwing clothes in the trash:\n1.Donate: Instead of throwing clothes away, consider donating them to local charities, thrift stores, or clothing banks. Many people can benefit from gently used clothing.\n2.Repair and repurpose: Extend the lifespan of your clothes by repairing them when they get damaged or giving them a new purpose through DIY projects, such as turning old t-shirts into rags or creating new items from old ones.\n3.Sell or swap: Consider hosting a clothing swap with friends or using online platforms to sell or exchange unwanted clothing items. This way, you can give your clothes a new home while reducing waste."}
   

    # Send a completion call to generate an answer
    
    image_detection_res = categorizeObject()
    method_of_disposal = image_detection_res[0]
    user_input = image_detection_res[1]

    if method_of_disposal == "recycling":
        user_q = {"role": "user", "content": "give me specific stats for recycling a {} in 5 bullet points".format(user_input)}
    elif method_of_disposal == "compost":
        user_q = {"role": "user", "content": "give me specific stats for composting a {} in 5 bullet points".format(user_input)}
    elif method_of_disposal == "landfill trash":
        user_q = {"role": "user", "content": "give me specific stats on the problem with throwing {0} in the trash in 3 bullet points and give me ways to avoid throwing {1} in the trash in 3 bullet points".format(user_input, user_input)}
    else: 
        user_q = {"role": "user", "content": "why special processing facility?"}

    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=[
            role,
            q1, a1,
            q2, a2,
            q3, a3,
            user_q
        ]
    )
    
    answer = response['choices'][0]['message']['content']
    print(answer)
    return answer

def giveImpactAndSuggestions():
    return giveImpactAndSuggestionsOpenai()


