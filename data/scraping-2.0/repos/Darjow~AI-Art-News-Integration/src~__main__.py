from openAI.chatgpt import ChatGPT
from openAI.dalle import DallE
from scraper.Scraper import Scraper



def main():
  scraped_data = Scraper().start_scraping()
  prompt = ChatGPT().start_new_conversation(scraped_data)
  image = DallE().generate_image(prompt)
  print(image)
  

  
if __name__ == "__main__":
  main()