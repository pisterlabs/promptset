import json
from openai import OpenAI
from Env import Env

class Get_Req_Yrs:

    def __init__(self):
        # self.data = data
        self.env = Env()

    def get_req_yrs(self):
       
        # setup var for writing
        data = {}

        # old_data = {}
        # with open('data.json') as f:
        #     old_data = json.load(f)


        # read data from json file
        with open('jobs_wth_desc.json') as f:
            data = json.load(f)
        

        # new jobs found 
        new_find = {}
        # error dict:
        error_dict = {}
        # final_data
        final_data = {}


        # Find required years of experience
        for key in data:

            if("required_years_of_experience" in data[key]):
                continue
            else:
                
                job_description = data[key]["job_desc"]
                # print(job_description)

                try:
                    client = OpenAI(
                        # defaults to os.environ.get("OPENAI_API_KEY")
                        api_key=self.env.CHATGPT_API_KEY,
                    )

                    required_years = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": f"how many years of experience are required (NOT preferred but required, please make sure you only send the required years; the preferred years is not the required years) for the following job, just give a single number, only respond with the number:\n {job_description}",
                            }
                        ],
                        model="gpt-3.5-turbo",
                        )
                        
                    try:
                        data[key]["required_years_of_experience"] = int(required_years.choices[0].message.content)

                        if(data[key]["required_years_of_experience"] < int(self.env.TOTAL_YEARS_OF_EXPERIENCE)+1):
                            final_data[key] = data[key]
                            new_find[key] = data[key]

                        print(data[key]["required_years_of_experience"])
                    
                    except Exception as e: 
                        data[key]["required_years_of_experience"] = f"__ERROR__: {str(e)}"
                        error_dict[key] = data[key]
                        print(f"__ERROR__: {str(e)}")
                except Exception as e: 
                    data[key]["__ERROR__OpenAI"] = f"__ERROR__: {str(e)}"
                    error_dict[key] = data[key]
                    print(f"__ERROR__OpenAI: {str(e)}")
                
                

        # all data
        with open('all_data.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


        # final data
        with open('final_data.json', 'w', encoding='utf-8') as file:
            json.dump(final_data, file, ensure_ascii=False, indent=4)


        # error data
        with open('error_openai.json', 'w', encoding='utf-8') as f:
            json.dump(error_dict, f, ensure_ascii=False, indent=4)

        # new data
        with open('new_find.json', 'w', encoding='utf-8') as f:
            json.dump(new_find, f, ensure_ascii=False, indent=4)
