from pandasai import SmartDataframe
import random
from pandasai.llm import OpenAI, Starcoder, GooglePalm
from pandasai.helpers.openai_info import get_openai_callback

greetings = ["Greetings, intrepid explorer of the data cosmos!",
"Hello, fellow data enthusiast! Welcome to the realm of data science!",
"Salutations, brave soul venturing into the data universe!",
"Hey there, data pioneer! Let's embark on a journey through the cosmos of data science!",
"Welcome, data voyager! Prepare to dive into the wonders of data science!",
"Greetings, data adventurer! Are you ready to explore the depths of data science?",
"Hello, data aficionado! Step into the world of data science and unlock its mysteries!",
"Salutations, data trailblazer! Your journey through data science begins now!",
"Hey, data explorer! Join me in unraveling the secrets of the data cosmos!",
"Welcome aboard, data traveler! Let's set sail into the vast seas of data science!",
"Hello, data visionary! Get ready to chart a course through the cosmos of data!",
"Greetings, data seeker! Your quest for knowledge in data science starts here!",
"Salutations, data prodigy! Time to delve into the marvels of data science!",
"Hey there, data maestro! Together, we'll orchestrate symphonies with data!",
"Welcome, data maestro! Prepare to conduct experiments in the lab of data science!",
"Hello, data luminary! Step into the limelight of data science brilliance!",
"Salutations, data virtuoso! The stage is set for your performance in data science!",
"Hey, data sage! The scrolls of data science wisdom await your perusal!",
"Greetings, data connoisseur! Savor the flavors of data science expertise!",
"Welcome, data guru! Your sanctuary of knowledge in data science awaits!",
"Greetings, data virtuoso! Brace yourself for a journey through the data cosmos!",
"Hello, data enthusiast! Ready to embark on a data-driven odyssey?",
"Salutations, data maestro! Let's compose symphonies with data together!",
"Hey there, data pioneer! The frontier of data science beckons!",
"Welcome, data explorer! Time to navigate the data galaxy!",
"Greetings, data voyager! Your spaceship to the data universe awaits!",
"Hello, data prodigy! Prepare to unlock the secrets of data alchemy!",
"Salutations, data connoisseur! Sip from the goblet of data wisdom!",
"Hey, data maven! Prepare to craft masterpieces with data brushes!",
"Welcome, data sorcerer! Ready to conjure insights from raw data?",
"Hello, data visionary! Your canvas in the art of data science is ready!",
"Greetings, data alchemist! Let's transmute raw data into gold!",
"Salutations, data sage! Your sanctuary of data wisdom awaits!",
"Hey there, data magician! Prepare to dazzle with data-driven feats!",
"Welcome, data sage! The scrolls of data knowledge await your touch!",
"Hello, data luminary! Your beacon of data enlightenment shines bright!",
"Greetings, data adept! Step into the dojo of data mastery!",
"Salutations, data oracle! The prophecies of data science await!",
"Hey, data guru! Ready to unlock the gates to data nirvana?",
"Welcome, data artisan! Let's sculpt insights from the data quarry!"]

class AumotativeReport:
    def __init__(self, sequencing_report, origin_report, global_params):
        self.sequencing_report = sequencing_report
        self.origin_report = origin_report
        self.global_params = global_params
        self.smart_report = None
        self.llm = self.get_api()
    
    def prepare_llm(self, ai_platform, api_key):
        if ai_platform == "openai":
            llm = OpenAI(api_token = api_key)
        elif ai_platform == "huggingface":
            llm = Starcoder(api_token=api_key)
        elif ai_platform == "GooglePalm":
            llm = GooglePalm(api_token=api_key)
        return llm
    

    def get_api(self):
        api_key = self.global_params["api"]
        if api_key == '':
            while True:
                print("Before you start to chat with your data, please make sure that you understand the environmental impact of large language models (LLMs).\nAlthouth there is a huge difference in training and using these models regarding the energy consumption, we need to be aware of the impact of our actions.\nEspecially if we use them for non-essential tasks.\nI hope you can help to create a rising awarness for this topic since it vanishes under the surface of the AI hype.\nFurther, always question the results of such language models, especially in data science,\nsince good and reproducible data is the origin of high quality research and new findings!")
                platform = input("Do you want to use OpenAI (1), HuggingFace (2) or Google Palm (3)?\nMake sure to add a corresponding payment method to your account.\nOtherwise the chat will not work.")
                if platform in ["1", "2", "3"]:
                    if platform == "1":
                        ai_platform = "openai"
                    elif platform == "2":
                        ai_platform = "huggingface"
                    elif platform == "3":
                        ai_platform = "google"
                    break
                else:
                    pass
            while True:
                api_key = input("Please enter your API key:\n")
                llm = self.prepare_llm(ai_platform, api_key)
                self.smart_report = SmartDataframe(self.sequencing_report, config={"llm": llm,  "conversational": False})
                try:
                    print("Test API key.\nTell me the shape of my dataframe")
                    with get_openai_callback() as cb:
                        response = self.smart_report.chat("Tell me the shape of my dataframe")
                        print(response)
                        print(cb)
                        break
                except:
                    print("API key not valid. Please try again.")
                    continue
        else:
            llm = self.prepare_llm(self.global_params["ai_platform"], api_key)
        return llm
    
    def chat_(self,report = "processed", conversation = False):
        if report == "processed":
            self.smart_report = SmartDataframe(self.sequencing_report, config={"llm": self.llm,  "conversational": False})
        else:
            self.smart_report = SmartDataframe(self.origin_report, config={"llm": self.llm,  "conversational": False})
        salud = random.choice(greetings)
        print(salud)
        user_question = input("How can I help you?\n")
        while True:
            with get_openai_callback() as cb:
                response = self.smart_report.chat(user_question)

                print(response)
                print(cb)
            if conversation == False:
                response = ""
                break
            else:
                user_question = input("What else can I do for you? Type 'bye' or 'exit' to leave.\n")
                if user_question in ["bye", "exit"]:
                    
                    break
                else:
                    continue
            
        return response 
                
