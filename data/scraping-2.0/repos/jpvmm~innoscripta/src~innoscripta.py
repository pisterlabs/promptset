import os

from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from serpapi import GoogleSearch
import logging
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

logger = logging.getLogger(__name__)

with open("external_data/google-countries.json", "r") as f:
    GL_COUNTRIES = json.load(f)

with open("external_data/google-languages.json", "r") as f:
    HL_COUNTRIES = json.load(f)

with open("external_data/google-domains.json", "r") as f:
    DOMAINS = json.load(f)


class Innoscripta:
    """Class of innoscript solution"""

    def __init__(self, name: str, country: str, website: str = None):
        """
        Initialize the Innoscripta search engine.

        Args:
            name (str): name of the company
            country (str): name of the country of the company
            website (str): website of the company
        """
        self.name = name
        self.country = country
        if not website:
            logger.info("Website not provided, searching for it....")
            self.website = self.find_website(company_name=name)
            logger.info(f"Found website! {self.website}")
        else:
            self.website = website
        self.llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def find_website(self, company_name) -> str:
        """
        Try to find the website of the company

        Args:
            company_name (str): The name of the company.
        Returns:
            website (str): website of the company
        """
        search = GoogleSearch(
            {
                "q": f"{company_name} + website",
                "engine": "google",
                "location": "Austin, Texas",
                "api_key": SERPAPI_KEY,
            }
        )
        response = search.get_dict()

        website = response["organic_results"][0]["link"]

        if website:
            return website
        else:
            return " "

    def main(self):
        """
        Will do the innoscripta querying
        """
        logger.info("Doing GPT search...")
        parsed_gpt_ouput = self.gpt_call()
        logger.info("Doing Google search...")
        google_query = self.google_query_formation(
            parsed_gpt_ouput["products_services"]
        )
        imgs = self.google_search(google_query)
        parsed_gpt_ouput["images"] = imgs

        return parsed_gpt_ouput

    def gpt_call(self) -> dict:
        """
        Will call gpt-3.5-turbo for querying informations about a company.

        Args:
            name(str): Name of the company
            country(str): Country of the company
            website(str): Website of the company

        Results:
            output(dict) = Parsed output of OpenAIAPI
        """

        prompt = self.prompt_template()

        chain = LLMChain(llm=self.llm, prompt=prompt)
        output = chain.run(
            {
                "name_of_company": self.name,
                "country_of_company": self.country,
                "website_of_company": self.website,
            }
        )
        parsed = self.parse_output(output)

        return parsed

    def google_query_formation(self, products: list) -> str:
        """
        Will manipulate strings to create Google search query

        Return:
            google_query(str): Google query
        """
        products_services_str = " + ".join(products)

        return " + ".join([self.name, products_services_str])

    def google_search(self, query: str) -> list:
        """
        Will query google for images based in the output of OpenAIAPI

        Args:
            query(str): Formated query using the output openaiAPI

        Results:
            imgs(list): List with URLs for images
        """
        gl = self.get_gl()
        hl = self.get_hl(gl)
        domain = self.get_google_domain()

        search = GoogleSearch(
            {
                "q": query,
                "engine": "google_images",
                "location": self.country,
                "api_key": SERPAPI_KEY,
                "google_domain": domain,
                "gl": gl,
                "hl": hl
            }
        )
        response = search.get_dict()
        imgs = [r.get("original", None) for r in response["images_results"][:10]]

        return imgs
    
    def get_gl(self):
        """
        Get Google Country to google image query
        """
        for d in GL_COUNTRIES:
            print(d["country_name"])
            if d["country_name"].lower() == self.country.lower():
                print(d["country_code"])
                return d["country_code"]
        print("country_code not found!")
        return "us"
    
    def get_hl(self, gl):
        """
        Get Google Languages to google image query
        """
        for d in HL_COUNTRIES:
            if d["language_code"] == gl:
                print(f"Language code: {d['language_code']}")
                return d["language_code"]
        print("HL not found!")
        return "en"
    
    def get_google_domain(self):
        """
        Get Googke Domain to google image_query
        """
        for d in DOMAINS:
            if d["country_name"] == self.country:
                print(f"Found domain {d['domain']}")
                return d["domain"]
        print("Domain not found!")
        return "google.com"

    def prompt_template(self):
        template = """
        I'll give you three inputs. These inputs will be the name of the company, 
        the country of the company, and the website company. The website of the company
        is not mandatory, so it can be just an empty string.
        If the website was not provided, gather all info you can with just name and country.
        You have to give me the products and services that the company offers as output.
        you dont need to give me nothing more than the ouput.

        input:
        IKEA Deutschland GmbH & Co. KG
        Germany
        ikea.com

        the output must be in this format, please use it:
        "products_services": Furniture, Home decor, Kitchen and Dining;
        "keywords":furniture, storage, lighting;
        "company_classification":5712 (Furniture Stores) – SIC, 442110 (Furniture Stores) – NAICS;
        "additional_informations":
            "Furniture" = "IKEA is well-known for its wide range of stylish and affordable furniture. They offer various furniture pieces for every room in the home, including living room, bedroom, kitchen, dining, and outdoor furniture. Their products feature modern designs, functionality, and often come in flat-pack form for easy transportation and assembly.",
            "Home Decor and Accessories" =  "In addition to furniture, IKEA provides a variety of home decor and accessories to enhance the style and functionality of living spaces. This includes items such as rugs, curtains, lighting fixtures, mirrors, frames, plants, and decorative storage solutions.",
            "Kitchen and Dining" = "IKEA offers a comprehensive range of kitchen and dining products. This includes kitchen cabinets, countertops, appliances, cookware, utensils, tableware, and dining furniture. They provide solutions for various kitchen styles, sizes, and budgets.",
            "Storage and Organization" = "IKEA specializes in storage and organization solutions to help keep homes tidy and efficient. They offer a wide selection of shelves, storage units, wardrobes, drawers, and closet systems. These products are designed to maximize space and provide smart storage solutions.",
            "Bedroom Furniture and Mattresses"= "IKEA provides bedroom furniture and mattresses that cater to different preferences and needs. They offer beds, bed frames, mattresses, dressers, wardrobes, and bedding accessories. Their products focus on comfort, functionality, and innovative design.",
            "Bathroom Furnishings" = "For bathrooms, IKEA offers a range of furnishings and accessories, including vanities, cabinets, sinks, faucets, showers, storage solutions, and bathroom textiles. These products aim to optimize space utilization and create a stylish and functional bathroom environment.",
            "Children's Furniture and Toys" = "IKEA features a variety of furniture and toys designed specifically for children. They offer children's beds, desks, storage systems, playroom furniture, toys, and decor items. Their products prioritize safety, durability, and imaginative play.",
            "Textiles and Fabrics" = "IKEA provides an assortment of textiles and fabrics for home decor, including curtains, blinds, rugs, cushions, bedding, and fabrics by the yard. They offer a wide selection of colors, patterns, and materials to suit different styles and preferences.",
            "Smart Home Solutions" = "In line with the growing trend of smart homes, IKEA offers smart home solutions such as lighting systems, wireless chargers, smart plugs, and integrated furniture with built-in technology. These products aim to enhance convenience, energy efficiency, and connectivity in the home.",
            "Home Delivery and Assembly Services" = "IKEA offers home delivery services to bring purchased products directly to customers' homes. They also provide assembly services, where IKEA's professionals can assemble the furniture and ensure it is ready to use.",


        do it yourself now.
        input:
        {name_of_company}
        {country_of_company}
        {website_of_company}

        what is the output?
        """

        prompt = PromptTemplate(
            input_variables=[
                "name_of_company",
                "country_of_company",
                "website_of_company",
            ],
            template=template,
        )

        return prompt

    def parse_output(self, output_langchain: str) -> dict:
        """
        Will parse the output_langchain of the Langchain query

        Args:
            output(str): The output_langchain of Langchain query.

        Returns:
            result_dict(dict): the parsed output_langchain to dict
        """

        result_dict = {}

        sections = [section.strip() for section in output_langchain.split(";")]

        for section in sections:
            if section:
                if "additional_informations" in section:
                    header = "additional_informations"
                    values = self.parse_additional_info(section)
                else:
                    header, values_str = section.split(":")
                    header = header.strip('"')
                    values = [value.strip() for value in values_str.strip("[]").split(",")]

                result_dict[header] = values

        return result_dict

    def parse_additional_info(self, data):
        # Parse to dict
        additional_informations = {}

        # Split into lines and iterate
        for line in data.split('\n'):
            # Ignore lines without '='
            if '=' not in line:
                continue

            # Split line into key-value pair
            key, value = line.split('=', 1)

            # Remove unwanted characters from key and value
            key = key.replace('"', '').strip()
            value = value.replace('"', '').strip()

            # Add to dictionary
            additional_informations[key] = value
        return additional_informations
