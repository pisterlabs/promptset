from openai import OpenAI
from dotenv import load_dotenv
from _brands_obj import BRANDS
import os
import pymongo
import json
from Colors import Colors
from Brand import Brand


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f'{Colors.CYAN}OPENAI_API_KEY: {OPENAI_API_KEY} {Colors.RESET}')
MONGO_URI = os.getenv("MONGO_URI")
print(f'{Colors.MAGENTA}MONGO URI: {MONGO_URI} {Colors.RESET}')

client = OpenAI()



# Function to insert brands into the database
def insert_brands():
    for brand_data in BRANDS:

        system_content = f"You are a knowledgeable assistant with comprehensive information about various companies and brands as well designed to output JSON.  For this and future requests we will be generating information about the company Warner Bros. Discovery and its subsidiary companies and brands.  This specific request will cover the subsidiary brand {brand_data['name']}.  We are building a database of information about the company and its brands so please provide detailed and accurate responses to the following questions, even if some details need to be estimated or inferred. If exact information is not available, offer the most likely or plausible details based on your extensive database of knowledge.  For the returned JSON data, please always use these specific keys: name, image, founding_year, founder, history, CEO, board_of_directors, number_of_employees, revenue_information, location, popular_brands_content, description."

        print(f"{Colors.YELLOW}Querying OpenAI for {brand_data['name']} data...",  f"{Colors.RESET}")
        
        # query openAI for the data
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={ "type": "json_object" },
            seed=42,
            messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"What year was {brand_data['name']} founded?"},
            {"role": "user", "content": f"Write a detailed biography of the history of {brand_data['name']}"},
            {"role": "user", "content": f"Who is the CEO of {brand_data['name']}?"},
            {"role": "user", "content": f"Who is on the board of directors for {brand_data['name']}?"},
            {"role": "user", "content": f"How many employees does {brand_data['name']} have?"},
            {"role": "user", "content": f"Can you provide any information about the company's revenue?"},
            {"role": "user", "content": f"Where is {brand_data['name']} located?"},
            {"role": "user", "content": f"What are some popular shows, or movies, podcasts or brands produced by {brand_data['name']}?"},
            {"role": "user", "content": f"Please write a brief description of {brand_data['name']}."},
            ],
        )

        for choice in response.choices:

            try:
                content = json.loads(choice.message.content)
            except json.JSONDecodeError:
                print(f"{Colors.RED}Failed to parse JSON content: {choice.message.content}",  f"{Colors.RESET}")
               
                continue
               # Assuming 'content' is now a dictionary
            print(f"{Colors.GREEN}content: {content}",  f"{Colors.RESET}")
              
            



            brand = Brand(
                name=brand_data.get("name", ""),
                image=brand_data.get("image", ""),
                founding_year=content.get("founding_year", ""),
                founder=content.get("founder", ""),
                history=content.get("history", ""),
                CEO=content.get("CEO", ""),
                board_of_directors=content.get("board_of_directors", ""),
                number_of_employees=content.get("number_of_employees", ""),
                revenue_information=content.get("revenue_information", ""),
                description=content.get("description", ""),
                location=content.get("location", ""),
                popular_brands_content=content.get("popular_brands_content", [])
            )
      
            print(f"{Colors.BLUE}brand: {brand.__str__()}",  f"{Colors.RESET}")
            
            
            
            # Save the brand to the database
            print(f"")
            print(f"{Colors.CYAN}Saving {brand.name} to database...",  f"{Colors.RESET}")

            # if brand already exists, skip it
            if Brand.get_all_brands() is not None:
                for existing_brand in Brand.get_all_brands():
                    if brand.name == existing_brand["name"]:
                        # compare the two brands and only update the fields that are empty
                        print(f"")
                        print(f"{Colors.MAGENTA}Brand {brand.name} already exists in database.  Skipping...",  f"{Colors.RESET}")
                        break
                else:
                    print(f"")
                    print(f"{Colors.GREEN}Brand {brand.name} does not exist in database.  Saving...",  f"{Colors.RESET}")
                    brand.save()
          
       








if __name__ == "__main__":
    insert_brands()
    print("All brands inserted successfully.")
    print(f"{Colors.GREEN}All brands inserted successfully.",  f"{Colors.RESET}")