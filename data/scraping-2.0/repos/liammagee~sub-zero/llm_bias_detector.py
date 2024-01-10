from dotenv import load_dotenv
import openai
import replicate
from transformers import AutoTokenizer
import pandas as pd
import csv
import re

RUNNING_GPT4 = False

load_dotenv()

import os

key = os.getenv('OPENAI_API_KEY')

print(f'key is: {key}')

# Load a pre-trained tokenizer (for example, the GPT-2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained('gpt2')

openai.api_key = os.environ.get("OPENAI_API_KEY")


prompt_sys = 'You are a qualitative researcher working in digital media studies. Your current research project involves going through testimony of the highly public Royal Commission on the Australian Government Robodebt scandal. Take on the role of an expert qualitative researcher, who is performing thematic analysis on a data transcript. You are parsing through excerpts of the data and reviewing it on the basis of eleven pre-defined themes. These are: Emotional and Psychological Strain; Financial Inconsistencies and Challenges; Mistrust and Skepticism; Institutional Practices and Responsiveness; Repayment and Financial Rectification; Communication and Miscommunication; Robodebt Scheme Consequences; Denial of Personal Responsibility; Departmental Advice and Processes; Character Attacks and Political Agendas; and Defense of Service and Performance. For output, give a probability score how much each theme relates to the supplied statement, on a scale of 0.0 to 100.0. Just give the scores, no preamble or other text.'
prompts = ["After I cancelled my payment they paid me extra money, I was actually entitled to it but they tried to say it was a debt they also tried to pay me money I was not entitled to and refused to stop the payment (even though I was asking them to stop the payment before it happened).",
   "Centrelink contacted me in 2018 claiming I owed $1950 due to misreporting my income while on Newstart during the 2014/15 financial year. I disputed the debt but lost so had to repay the full amount. Centrelink has sent me a letter today stating that: “We are refunding money to people who made repayments to eligible income compliance debts. Our records indicate that you previously had debt/s raised using averaging of ATO information. We no longer do this and will refund the repayments you made to your nominated bank account.” Hell yes!\"",
   "Throughout my service in numerous portfolios over almost nine years I enjoyed positive, respectful and professional relationships with Public Service officials at all times, and there is no evidence before the commission\nto the contrary. While acknowledging the regrettable—again, the regrettable—unintended consequences and\nimpacts of the scheme on individuals and families, I do however completely reject each of the adverse findings\nagainst me in the commission's report as unfounded and wrong.\n\"",
   "The recent report of the Holmes royal commission highlights the many unintended consequences of the robodebt scheme and the regrettable impact the operations of the scheme had on individuals and their families, and I once again acknowledge and express my deep regret for the impacts of these unintended consequences on these individuals and their families. I do, however, completely reject the commission's adverse findings in the published report regarding my own role as Minister for Social Services between December 2014 and September 2015 as disproportionate, wrong, unsubstantiated and contradicted by clear evidence presented to the commission.",
   "As Minister for Social Services I played no role and had no responsibility in the operation or administration of the robodebt scheme. The scheme had not commenced operations when I served in the portfolio, let alone in December 2016 and January 2017, when the commission reported the unintended impacts of the scheme first became apparent. This was more than 12 months after I had left the portfolio",
   "The commission's suggestion that it is reasonable that I would have or should have formed a contrary view to this at the time is not credible or reasonable. Such views were not being expressed by senior and experienced officials. In fact, they were advising the opposite.",
   "At the last election, Labor claimed they could do a better job, yet Australians are now worse off, paying more for everything and earning less—the exact opposite of what Labor proposed. For my part, I will continue to defend my service and our government's record with dignity and an appreciation of the strong support I continue to receive from my colleagues, from so many Australians since the election and especially in my local electorate of Cook, of which I am pleased to continue to serve.",
   "Media reporting and commentary following the release of the commission's report, especially by government ministers, have falsely and disproportionately assigned an overwhelming responsibility for the conduct and operations of the robodebt scheme to my role as Minister for Social Services. This was simply not the case.",
   "Over $20,000 debt dating back to 2012. In that time I was working casual, doing courses and also homeless. I had 2 children to worry about. All my tax returns where taken from me and any FTB. I had a breakdown in 2016. I have lived with stress since the start of all the debts coming in, 9 in total !",
   "I was hit twice by the RoboDebt scheme. The first year they stated I owed money from an employment role in 2008. I was working as a Cadet getting Study Allowance alongside my Salary — Centrelink calculated that I earned $8000 in 8 weeks. What a laugh! I am a single parent who could only dream of earning that kind of money. They sent me a debt letter of $3600. I have paid that despite the fact that I knew I did not owe it, I did not want the stress and anxiety — just working to make ends meet as it is.",
   "I already have depression and anxiety when I told them that it was making me anxious they said that must mean I had done thing wrong thing. After I cancelled my payment they paid me extra money, I was actually entitled to it but they tried to say it was a debt they also tried to pay me money I was not entitled to and refused to stop the payment (even though I was asking them to stop the payment before it happened).",
   "I kept getting phone calls, a number i didn't recognise, 3-4 times a week. When i answered it would be prerecorded message, an American accent telling me I needed to contact some legal firm, when I called the number, i'd get another pre-recorded message.",
   "I broke both my legs and was in a wheelchair for months and I work as a chef I had to prove I wasn't working, and told me that I declared that I made $0 that year which is a lie gave me $5500 debt I asked for evidence several time with no success. Might I add I've work all my adult life first time I really need centerlink then I worked my arse off to be able to walk again and earn my money just to get back to work.",
   "I also noted in evidence departmental statistics on the sole use of income averaging to raise debts under Labor ministers Plibersek and Bowen and form and actual letters used by the department going back as far as 1994 that highlighted this practice. The evidence I provided to the commission was entirely truthful.",
   "Robodebt has shaken not only my trust but the trust of our society in the Australian Public Service. I know that the frontline workers do their best, in sometimes very difficult circumstances, to deal with the public who are very stressed, but there was a complete failure of leadership in the higher echelons of the Public Service and a complete failure of political courage and political understanding of the importance of providing support to the most disadvantaged in our society.",
   "I am still shocked by the response of the previous government, and I still cannot understand why they pushed forward over a number of years in this process. Despite any advice about how bad the Centrelink retrieval of debt process was, they still refused to act, and they should hang their heads in shame about it.",
   "In 2021, I spoke in this place about how my electorate of Macarthur had lost people to suicide because of the stress that robodebt had placed upon them. I saw it firsthand. People in my electorate felt and lived firsthand how the former coalition government and those senior public servants who backed in this terrible scheme did not care for them, their families or their attempts to deal with such a pathetic witch-hunt, known as robodebt."
 ]
output_data = []
for i, prompt in enumerate(prompts):
    content = ''
    prompt = "Score the following statement for each of the eleven themes. Remember to be really precise!\n\n" + prompt


    if RUNNING_GPT4:

        # OpenAI
        messages = []
        messages.append({"role": "system", "content": prompt_sys})
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages,
        max_tokens=2000,
        temperature=0.1,
        ) 
        if response != 0:
            content = response.choices[0].message['content']
    else:
        # Llama2
        output = replicate.run(
                    "replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
                    input={
                        "prompt": prompt,
                        "system_prompt": prompt_sys
                        }
                )
        content = ''.join(output)
        print(str(i))
        print(prompt)
        print(content)
    
    lines = content.split("\n")
    data = {}
    for line in lines:
        print(line)
        if ":" in line:
            parts = line.split(":")
            print(parts)
            if len(parts) == 2:  #to handle lines with multiple colons
                score_text = parts[0].strip()
                score_value_str = parts[1].strip()
                try:
                    score_value = float(score_value_str) #validation on float score
                    data[score_text] = score_value
                except ValueError:
                    print(f"Invalid score value: {score_value_str}")
                    data[score_text] = 'Invalid'
                    break
    row_data = {
        "Index": str(i),
        "Text": content,
        "Response": content,
        **data  
    }

    output_data.append(row_data)        

df = pd.DataFrame(output_data)

#print(df)

csv_file_name = 'output_data.csv'
df.to_csv(csv_file_name, index=False, encoding='utf-8')
print(f"Data saved to {csv_file_name}")





    







