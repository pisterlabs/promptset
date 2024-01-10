import os
import pandas as pd
import openai
import pinecone
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.agents import create_csv_agent
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Get the current directory
current_dir = os.path.dirname(__file__)

# Construct the file path
file_path = os.path.join(current_dir, 'responsive-ads-2023-04-03.csv')


load_dotenv()


# Initialize Pinecone Connection
pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.environ.get("PINECONE_ENVIRONMENT")  # next to api key in console
)

llm = OpenAI(temperature=0)
index_name = os.environ.get('PINECONE_INDEX')
embeddings = OpenAIEmbeddings()


class LangchainService:
    def __init__(self):
        self.csv_agent = create_csv_agent(OpenAI(temperature=0), file_path , verbose=True)
        self.df = pd.read_csv(file_path)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        pass

    def parse_website(self, url):
        """
        Parses the content of a website.

        Args:
            url (str): The URL of the website.

        Returns:
            str: The parsed content of the website.
        """
        try:
            print(f"Parsing website {url} starts")
            loader = WebBaseLoader(url)
            data = loader.load()
            print(f"Parsing website {url} ends")
            return data
        except openai.error.RateLimitError:
            return "Facing Some Problem in Parsing. Please try later"
        except Exception as ex:
            return ex
    
    def summarization(self, data):
        """
        Performs summarization on the given data.

        Args:
            data (str): The data to be summarized.

        Returns:
            str: The summarized content.
        """
        try:
            docs = self.text_splitter.split_documents(data)
            chain = load_summarize_chain(OpenAI(temperature=0), chain_type="map_reduce")
            summary = chain.run(docs)
            return summary
        except openai.error.RateLimitError:
            return "Facing Some Problem in Summarization. Please try later"
        except Exception as ex:
            return ex
        
    def show_winning_ads(self, keyword_name):
        """
        Performs summarization on the given data.

        Args:
            data (str): The data to be summarized.

        Returns:
            str: The summarized content.
        """
        try:
            query = f"Filter all rows where ad_group_ad.ad.responsive_search_ad.headlines column contain a substring {keyword_name}, then sorts those rows based on a RankWeight column in descending order, and finally selects at most 5 rows with the highest values in that column."
            result = self.csv_agent.run(query)
            return result
        except openai.error.RateLimitError:
            return "Facing Some Problem in Summarization. Please try later"
        except Exception as ex:
            return ex
    
    def get_winning_ads(self, keyword):
        """
        Retrieves similar ads based on the provided keyword.

        Args:
            keyword (str): The keyword used to filter similar ads.

        Returns:
            pandas.DataFrame: A DataFrame containing the similar ads.

        Raises:
            openai.error.RateLimitError: If there is a rate limit issue.
            Exception: If any other error occurs.
        """
        try:
            # Filter all rows where 'ad_group_ad.ad.responsive_search_ad.headlines' column contains a substring
            substring = keyword
            filtered_rows = self.df[self.df['ad_group_ad.ad.responsive_search_ad.headlines'].str.contains(substring)]
            # Sort the filtered rows by a specific column in descending order
            sorted_rows = filtered_rows.sort_values('RankWeight', ascending=False)
            # Select at most 5 rows with the highest values in the specific column
            result = sorted_rows.head(5)
            combined_rows = []
            for index, row in result.iterrows():
                headline = row['ad_group_ad.ad.responsive_search_ad.headlines']

                headlines = headline.split(";")
                headlines = headlines[:3]
                formatted_headlines = ", ".join([f"Headline {i+1}: \"{headline}\"" for i, headline in enumerate(headlines)])

                description = row['ad_group_ad.ad.responsive_search_ad.descriptions']
                combined_rows.append({'headline': formatted_headlines, 'description': f"Description: {description}"})
            return combined_rows
        except openai.error.RateLimitError:
            return "Facing Some Problem in retrieving similar ads. Please try again later."
        except Exception as ex:
            return ex

class OpenAIService:
    def __init__(self):
        pass
    
    def __get_completion(self, prompt, model="gpt-3.5-turbo", temperature=0):
        """
        Private method to interact with the OpenAI Chat API and generate completions.

        Args:
            prompt (str): The prompt for generating the completion.
            model (str): The name of the model to use (default: "gpt-3.5-turbo").
            temperature (float): The degree of randomness in the output (default: 0).

        Returns:
            str: The generated completion.
        """
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message["content"]
    
    def run_step_1(self, summary):
        """
        Runs step 1 of the process.

        Args:
            summary (str): The summary for step 1.

        Returns:
            str: The generated response for step 1.
        """
        prompt = f"""Based on the summary delimited by ```
          Does this business sell products, provide services or something else?  Please answer with only one word.
         “Product”, “Service” or  “Something else”
          summary: ```{summary}```
         """
        try:
            print(f"Going to call OpenAI, step 1: Start")
            response = self.__get_completion(prompt, temperature=0)
            print(f"Going to call OpenAI, step 1: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)


    def run_step_2(self, summary):
        """
        Runs step 2 of the process.

        Args:
            summary (str): The summary for step 2.

        Returns:
            str: The generated response for step 2.
        """
        prompt = f"""Based on the summary delimited by ```
          Is it in an industry that is potentially sensitive such as healthcare, drug rehab, funeral care, 
          counseling, or other industry that is likely to be used by somebody in a state of distress?  
          If so, please respond with a single word, “Sensitive” or “Non-sensitive” as appropriate.
          summary: ```{summary}```
         """
        try:
            print("Going to call OpenAI, step 2: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 2: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)

    def run_step_3(self, summary):
        """
        Runs step 3 of the process.

        Args:
            summary (str): The summary for step 3.

        Returns:
            str: The generated response for step 3.
        """
        prompt = f"""Based on the summary delimited by ```
          What noun do you think would best describe an ideal customer for this business?  
          For example, a dentist would refer to their customers as patients.  
          A lawyer would use the word client.  
          What noun would you use for this business? Just answer in one word.
          summary: ```{summary}```
         """
        try:
            print("Going to call OpenAI, step 3: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 3: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)

        
    def run_step_4(self, summary):
        """
        Runs step 4 of the process.

        Args:
            summary (str): The summary for step 4.

        Returns:
            str: The generated response for step 4.
        """
        prompt = f"""Based on the summary delimited by ```
        What is the name of this business?  You can ignore any legal status such as llc, ltd, or inc.
        Answer in this format: The name of the business is [Business name].
        summary: ```{summary}```
        """
        try:
            print("Going to call OpenAI, step 4: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 4: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
    
    def run_step_5(self, content, service_or_product):
        """
        Runs step 5 of the process.

        Args:
            content (str): The content for step 5.
            service_or_product (str): The service or product type.

        Returns:
            str: The generated response for step 5.
        """
        prompt = f"""Based on the content delimited by ```
        Please write a simple numbered list of the {service_or_product} that this business offers. 
        Your description of each [service] should be short and simple.  
        If you find more than 10 {service_or_product}, only list the 10 most important ones.
        content: ```{content}```
        """
        try:
            print("Going to call OpenAI, step 5: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 5: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)


    def run_step_6(self, content, service_or_product):
        """
        Runs step 6 of the process.

        Args:
            content (str): The content for step 6.
            service_or_product (str): The service or product type.

        Returns:
            str: The generated response for step 6.
        """
        prompt = f"""Based on the content delimited by ```
        Does this content highlight one {service_or_product} in particular or is it more general promoting their business as a whole.
        If it is a single {service_or_product} focus, you should respond with only the number from your list above,
        or if it is general, respond with the word “general” only. 
        No explanation is required in your response.
        content: ```{content}```
        """
        try:
            print("Going to call OpenAI, step 6: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 6: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
    
    def run_step_7(self, content): 
        """
        Runs step 7 of the process.

        Args:
            content (str): The content for step 7.

        Returns:
            str: The generated response for step 7.
        """
        prompt = f"""Based on the content delimited by ```
        Wonderful. Now let’s extract more information from this content and make a single list containing 5 types of element.
        The list should be in the format "Element: List item". You should not include headings in the list.
        All list entries should be simple and brief.

        Contacts:
        How can a visitor to this page make contact with the business? List each option available to the user that you have found on the page.

        Offer:
        Look for any special offers, discounts, or promotions on the page.

        Guarantee:
        Can you find a mention of a guarantee or warranty? If so, list it starting each item with “Guarantee:”

        Price:
        Can you find any prices mentioned? If so, list it starting with “Price:”

        USP:
        Look for any information, selling points, or promises that are made on the page that the business might use to set it apart from its competition.
        List these starting each item with “USP:”

        Call to Action:
        What phrases does the business use to encourage the user to take an immediate next step such as "get a quote now" or similar.

        content: ```{content}```
        """
        try:
            print("Going to call OpenAI, step 7: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 7: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
    
    def run_step_8(self, noun, service_or_product):
        """
        Runs step 8 of the process.

        Args:
            noun (str): The noun that describes the ideal customer for the business.
            service_or_product (str): The service or product offered by the business.

        Returns:
            str: The generated response for step 8.
        """
        prompt = f"""
        List 5 to 10 fears and worries that a {noun} is likely to experience when selecting a vendor for this {service_or_product}.
        """
        try:
            print("Going to call OpenAI, step 8: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 8: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
        
    def run_step_9(self, noun, service_or_product):
        """
        Runs step 8 of the process.

        Args:
            noun (str): The noun that describes the ideal customer for the business.
            service_or_product (str): The service or product offered by the business.

        Returns:
            str: The generated response for step 8.
        """
        prompt = f"""
        How do you think a {noun} buying this {service_or_product} hopes to feel when the service is delivered to a very high standard. List the top emotions and feelings you might expect. 
        """
        try:
            print("Going to call OpenAI, step 9: Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Going to call OpenAI, step 9: End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
        
    def run_step_10(self, noun, search_term, competitors_ads):
        """
        Runs step 10 of the process.

        Args:
            noun (str): The noun that describes the ideal customer for the business.
            search_term (str): The search term used by homeowners on Google.
            competitors_ads (str): The ads from the top competitors shown in Google search results.

        Returns:
            str: The generated response for step 10.
        """
        # Prepare the prompt for Step 10
        prompt = f"""
        The business advertises using Google Ads. The goal of an ad is to get a {noun} to click their ad and not their competitor ad by writing an ad that emphasizes the benefits of the business in comparison to its competitors.

        Google Ads are in the format:
        Headline 1 - Headline 2 - Headline 3 (optional)
        Description 1. Description 2 (optional)

        I have listed below 5 ads from their top competitors shown when a homeowner has typed "{search_term}" into Google search.
        Please read them and rank them from 1 (most likely that a {noun} would click) to 5 (least likely a homeowner would click).

        {competitors_ads}
        """

        try:
            print("Calling OpenAI API: Step 10 - Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Calling OpenAI API: Step 10 - End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
    
    def run_step_11(self, winning_ads):
        """
        Runs step 11 of the process.

        Args:
            winning_ads (str): A string containing the winning ads that have been successful in the industry for other advertisers.

        Returns:
            str: The generated response for step 12.
        """
        # Prepare the prompt for Step 12
        prompt = f"""
        I will now provide you with ads that have been successful in this industry for other advertisers delimited by ```.  
        You should consider this in the subsequent tasks.  Acknowledge with "ok".
        ```{winning_ads}```

        """

        try:
            print("Calling OpenAI API: Step 11 - Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Calling OpenAI API: Step 11 - End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)

    
    def run_step_12(self, business_name, content, industry):
        """
        Runs step 12 of the process.

        Args:
            business_name (str): The name of the business.
            content (str): The page content for the business.

        Returns:
            str: The generated response for step 11.
        """
        # Prepare the prompt for Step 11
        prompt = f"""
        Now using all your knowledge about {business_name}, their page content:```{content}``` , ads that have succeeded for other advertisers and competitor ads, you are to take on the role of Google Ads copywriter.  You are highly experienced and understand how to use emotion, urgency, calls to action, and other direct response mechanisms.  Your job is to write ad copy with the goal of being more likely to get the homeowner to click than the competitor ad you ranked number 1 earlier.  You should use your understanding of {industry} fears and worries, along with their hopes for success to make the copy emotionally engaging.

        The ads you create will be shown to the user in the format:
        Headline 1 - Headline 2 - Headline 3 (optional)
        Description 1. Description 2 (optional)

        Each headline should be between 20 and 30 characters long.

        Each description should be between 70 and 90 characters long.

        Write 12 headlines each between 20 and 30 characters long.  Each headline should focus on different attributes of Mighty Dogs service.

        Write 5 descriptions each of between 70 and 90 characters to support these headlines and further encourage the homeowner to click.  Again, this should not repeat the headline content but add to it.  You cannot exceed the character limits for Headlines and Descriptions in your output.
        """

        try:
            print("Calling OpenAI API: Step 12 - Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Calling OpenAI API: Step 12 - End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)


    
    def run_step_13(self, keyword):
        """
        Runs step 13 of the process.

        Args:
            keyword (str): The keyword to be used in the generated headlines.

        Returns:
            str: The generated response for step 12.
        """
        # Prepare the prompt for Step 12
        prompt = f"""
        Fantastic. You now have 4 tasks.
        1. Write 3 more headlines focusing on a [Special Offer].
        2. Write 3 headlines that focus on their [USP].
        3. Write 3 headlines that focus on [Testimonials].
        4. Write 3 headlines that use Dynamic Keyword Insertion in the format {keyword}.

        All headlines must use Title Case and not exceed 30 characters except for task 4, where 45 characters are permitted.
        """

        try:
            print("Calling OpenAI API: Step 13 - Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Calling OpenAI API: Step 13 - End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)
    
    def run_step_14(self):
        """
        Runs step 14 of the process.

        Returns:
            str: The generated response for step 13.
        """
        # Prepare the prompt for Step 13
        prompt = """
        Finally, write 3 headlines that use alliteration to emphasize the benefit.
        """

        try:
            print("Calling OpenAI API: Step 14 - Start")
            response = self.__get_completion(prompt, temperature=0)
            print("Calling OpenAI API: Step 14 - End")
            return response
        except openai.error.RateLimitError:
            return "Facing rate limit quota issue. Please try again later."
        except Exception as ex:
            return str(ex)



    
if __name__ == "__main__":

    langchain_service = LangchainService();
    openai_service = OpenAIService();

    # Step 1
    website_content = langchain_service.parse_website("https://ambassadorac.com/south-florida-plumbing-services/")
    summary = langchain_service.summarization(website_content)
    print(summary)
    step_1_result = openai_service.run_step_1(summary)
    service_or_product = step_1_result
    print(step_1_result)


    # Step 2
    step_2_result = openai_service.run_step_2(summary)
    print(step_2_result)

    # Step 3
    step_3_result = openai_service.run_step_3(summary)
    noun = step_3_result
    print(step_3_result)

    # Step 3
    step_4_result = openai_service.run_step_4(summary)
    business_name = step_4_result
    print(step_4_result)

    # Step 5
    step_5_result = openai_service.run_step_5(website_content,service_or_product)
    print(step_5_result)

    #Step 6
    landing_page_website_content = langchain_service.parse_website("https://ambassadorac.com/south-florida-plumbing-services/")
    summary_lading_page = langchain_service.summarization(landing_page_website_content)
    step_6_result = openai_service.run_step_6(landing_page_website_content,service_or_product)
    print(step_6_result)

    #Step 7
    step_7_result = openai_service.run_step_7(landing_page_website_content)
    print(step_7_result)

    #Step 8
    step_8_result = openai_service.run_step_8(noun,service_or_product)
    print(step_8_result)

    #Step 8
    step_9_result = openai_service.run_step_9(noun,service_or_product)
    print(step_9_result)

    #Step 10
    #Change the keyword, based on the keyword parameter it will find ads in xl sheets
    #Change the search term
    keyword_name = "Property Management"
    search_term = "Property Management"

    #Add competitors ads
    competitors_ads = []
    step_10_result = openai_service.run_step_10("client",search_term, competitors_ads)
    print(step_10_result)

    #For showing the rows coming from xl sheets
    langchain_service.show_winning_ads(keyword_name)

    #Getting the rows in actual formate need to send int the prompt 10
    get_winning_ads = langchain_service.get_winning_ads(keyword_name)
    winning_ads = ''
    for index, ads in enumerate(get_winning_ads):
        winning_ads += f"Ads:{index + 1} {ads['headline']}\n{ads['description']}\n"

    print(winning_ads)

    #Step 11
    step_11_result = openai_service.run_step_11(winning_ads)
    print(step_11_result)

    #Step 12
    industry = 'Property Management'
    step_12_result = openai_service.run_step_12(business_name,summary_lading_page,industry)
    print(step_12_result)

    #Step 13
    step_13_result = openai_service.run_step_13("{{KeyWord:{keyword_name}}}")
    print(step_13_result)

    #Step 14
    step_14_result = openai_service.run_step_14()
    print(step_14_result)

