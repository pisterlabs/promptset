import openai
import os
import json
from time import perf_counter
from pathlib import Path


def get_seo_text():
    
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    """ with open('../output/seo_template.json', 'r') as json_file:
        seo_template = json.load(json_file) """
        
    file_path = Path('../files/seo/post_linkedin.txt')
        
    """ with open(file_path, 'r') as txt_file:
        #html_template = html_file.read()
        text = txt_file.read() """
        
    
    
    city = 'Milan'
        
    ct = 'https://cheaptrip.guru/en-US/#/search/myPath/logo'
    
    #prompt = f'Act as full stack developer with 10+ years of experience. Create seo-targeted web landing page from the source: {seo_template}. Also insert redirect to {ct} after 5 sec relay.'
    """ prompt = f'''Act as seo expert with 10+ years of experience. Create seo-frendly web page for city: {city} by this example: {html_template}.
                Find out in {city} 10 'Free and Low-Cost Attractions' and add those web-links to corresponding content item as a list. 
                Also find out to 5 locations of 'Cheap Eats' in {city} and add those web-links to corresponding content item as a list.
                Verify those all web-pages actually exist.
                Change structure only in order to add those unnumbered lists.
                ''' """
    
    #print(prompt)
    """ response = openai.Completion.create(model="text-davinci-003",
                                        prompt=prompt,
                                        temperature=0,
                                        max_tokens=2500) """
                                  
    """ response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[
                                                        #{"role": "system", "content": "You are a social media manager."},
                                                        #{"role": "user", "content": f"Paraphrase the following text: {text}"}
                                                        {"role": "system", "content": "You are a seo manager."},
                                                        {"role": "user", "content": f"Extract keywords from this text: {text}"}
                                                    ]
                                            ) """

    response_img = openai.Image.create(
                                        #prompt=response['choices'][0]['message']['content'],
                                        prompt="bus on the road, blue sky, label on the road: pay less, visit more",
                                        n=3,
                                        size="1024x1024"
                                        
                                        )
    
    
    with open(file_path, 'a') as txt_file:
        #html_file.write(response['choices'][0]['text'])
        #txt_file.write("After paraphrasing\n")
        #txt_file.write(response['choices'][0]['message']['content'])
        for item in response_img['data']:
            txt_file.write(f"\nImage url:{item['url']}")
    #print(response['choices'][0]['text'])
    
    
    
if __name__ == '__main__':
    start = perf_counter()
    get_seo_text()
    print(perf_counter() - start)
    pass