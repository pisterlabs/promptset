from langchain.prompts import PromptTemplate;
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from loader import Loader
class GPTLink:
    def __init__(self):
        self.example1 = Loader("https://www.ebay.com/b/VMAXTANKS-Rechargeable-Batteries/48619/bn_7114644579")
        self.example1.extract()
        self.example2 = Loader("https://www.ebay.com/b/Windows-Vista-PC-Laptops-Netbooks/177/bn_2767810")
        self.example2.extract()
        self.example3 = Loader("https://www.ebay.com/b/CHANEL-Eyeglass-Frames/180957/bn_8430684")
        self.example3.extract()
        self.llm = ChatOpenAI(temperature=.7,model="gpt-4-1106-preview") 
        #self.llm = OpenAI(temperature=.7,model="gpt-3.5-turbo-1106")
        fewShotTemplate = """
            You are an SEO expert that creates a compelling description for a web page.  You use the information on
            the page to create this description.  You should not use the meta descriptions within the <head> tag to generate this
            description.  The content <h1>, <h2>, <h3> tags are particularly important.  Keep the descriptions under 
            160 characters.
                                
            Here are some examples of good descriptions:
                                
            HTML: {firstExtraction}
            Description: Find great deals for VMAXTANKS Rechargeable Batteries at the lowest prices online. 
                Choose by amp hours like 100 Ah, 10 Ah, 55 Ah &amp; more to find exactly what you need. 
                Free shipping for many items!
                                
            HTML: {secondExtraction}
            Description: Save Big on new &amp; used Windows Vista PC Laptops &amp; Netbooks from top brands like HP, Dell, ACER &amp; more. Shop our extensive selection of products and best online deals. Free Shipping for many items!
                                
            HTML: {thirdExtraction}
            Description: Capture great deals on stylish CHANEL Eyeglass Frames at the lowest prices. Choose by frame shape like Full Rim, Round, Cats Eye &amp; more to complete your look. Free shipping for many items!
"""
        self.fewShot = fewShotTemplate.format(firstExtraction=self.example1.extraction, secondExtraction=self.example2.extraction, thirdExtraction=self.example3.extraction)
        print("Few shot prompt")
        print(self.fewShot)
        print("-------------------------------------------------")
        self.prompt = PromptTemplate.from_template(self.fewShot + """
            Give a description for this HTML:
            HTML: {url}
            Description:
                                                        """)
    def createDescriptions(self, urls):
        for url in urls:
            loader = Loader(url)
            loader.extract()
            print("-------------------------------------------------")
            answerChain = LLMChain(llm=self.llm, prompt=self.prompt)
            html = Loader(url)
            html.extract()
            response = answerChain.run(html.extraction)
            print(response)

def main():
    print("Hello World!")
    gptLink = GPTLink()
    urls = [
    "https://www.ebay.com/b/Clothing-Shoes-Accessories/11450/bn_1852545",\
    "https://www.ebay.com/b/Designer-Handbags/bn_7117629183",\
    "https://www.ebay.com/b/Balenciaga-Bags-Handbags-for-Women/169291/bn_724671",\
    "https://www.ebay.com/b/Auto-Parts-Accessories/6028/bn_569479",\
    "https://www.ebay.com/b/Commercial-Truck-Air-Conditioning-Heating-Components/184825/bn_115384977",\
    "https://www.ebay.com/b/BMW-Car-and-Truck-Tail-Lights/33716/bn_580096",\
    "https://www.ebay.com/b/BMW-Car-and-Truck-Turn-Signals/33717/bn_578781",\
    "https://www.ebay.com/b/Automotive-Paint-Supplies/179429/bn_1880778",\
    "https://www.ebay.com/b/Collectible-Sneakers/bn_7000259435",\
    "https://www.ebay.com/e/fashion/nike-kobe-6-protro-reverse-red",\
    "https://www.ebay.com/b/PUMA-Sneakers-for-Men/15709/bn_58992",\
    "https://www.ebay.com/b/Mens-Sandals/11504/bn_57786",\
    "https://www.ebay.com/b/Luxury-Watches/31387/bn_36841947",\
    "https://www.ebay.com/b/Rolex-Watches/31387/bn_2989578",\
    "https://www.ebay.com/b/Rolex-Submariner-Watches/31387/bn_3001215",\
    "https://www.ebay.com/b/Minivan-Cars-and-Trucks/6001/bn_55180494",\
    "https://www.ebay.com/b/Pontiac-GTO-Cars/6001/bn_24016910",\
    "https://www.ebay.com/b/Collectible-Figures-Bobbleheads/149372/bn_3017826",\
    "https://www.ebay.com/b/Star-Wars-Collectible-Figures-Bobbleheads/149372/bn_93507041",\
    "https://www.ebay.com/b/Hunting-Scopes-Optics-Lasers/31710/bn_1865309",\
    "https://www.ebay.com/b/Hunting-Rifle-Scopes/31714/bn_1865568",\
    "https://www.ebay.com/b/PXI-VXI-Systems/181963/bn_16565561",\
    "https://www.ebay.com/b/Biodiesel-Equipment/159694/bn_16562059"]
    gptLink.createDescriptions(urls)
if __name__ == "__main__":
    main()
