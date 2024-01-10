from dotenv import load_dotenv
import openai
import os
load_dotenv()

api_key = os.getenv("API_KEY7")

def getsearch(solution):
    gptInput = f"convert the paragraph: {solution} into a single search phrase with less than 5 keywords to research information about the proposed solution. Make sure it is less than 5, or else."
    openai.api_key = api_key
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user", "content":gptInput}]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content



if __name__ == "__main__":
    inp = "The proposed solution is a garment rental service. Such a service should work closely with major clothing brands. When buying items, customers should have the option to buy or rent. This model would be like a subscription service where customers can select a set number of items each month, wear them, then return them for other items. Clothing would be professionally cleaned between customers and repaired as necessary to maximize its life cycle. When a garment is no longer suitable for rental, it can be recycled into new clothes. This solution reduces the number of garments produced, the amount of transportation needed, and the quantity of clothes going to landfills. Completely damaged or unusable textiles can be reused or recycled into new products. It also gives financial value to businesses as it transforms fashion from a single-purchase model into a subscription service, creating a continuous income stream. Its feasibility and scalability depend on factors such as location, culture, and income level. However, as digital platforms become more common for commerce, this concept could be globally implemented."
    getsearch(inp)
