import os
import openai
import pandas as pd
from gpt_3_fine_tuning.semantic_search import search_takeaways, merge_dataframes_3
from dotenv import load_dotenv

messages_data = []
messages = [
        {"role": "system", "content": "You are an AI assistant for the University of Nicosia (UNIC). You provide answers"
                                      " to user inquiries using the data provided. Use a friendly and professional tone."},
        {"role": "user", "content": "Data:\nUniversity of Nicosia\n-  Located in the Republic of Cyprus \n- Offers "
                                    "students the ability to study from a variety of disciplines \n- Has a strong "
                                    "online presence \n- Provides an excellent learning environment \n- Has "
                                    "experienced faculty and staff who are dedicated to helping students reach their "
                                    "academic goals \n- Offers a variety of support services, including email and "
                                    "phone communication\nUniversity of Nicosia\n-  Offers a wide range of courses\n- "
                                    "Has a great reputation in the education sector\n- Has a well-equipped campus\n- "
                                    "Provides a safe and secure environment for students\n- Has a supportive and "
                                    "dynamic faculty\n- Offers excellent student services\n- Has a variety of "
                                    "educational resources available\n- Is dedicated to the health and safety of its "
                                    "students, providing nutritious meals and preventing the spread of viruses and "
                                    "bacteria\nUniversity of Nicosia\n-  Established university with a positive track "
                                    "record\n-  Comprehensive range of courses available at undergraduate and "
                                    "postgraduate level\n-  Dedicated support staff to assist with applications\n-  "
                                    "Variety of study options available, including online learning\n-  Ability to "
                                    "transfer applications between members of the same family."
                                    "\n\nUsing the data reply to the following message:\n\nMessage: What is the "
                                    "University of Nicosia?"},
        {"role": "assistant", "content": "Hello! The University of Nicosia (UNIC) is the largest university in "
                                         "Southern Europe that teaches primarily in English. It was established in "
                                         "1980 and is a leading private institution in Cyprus. UNIC has "
                                         "state-of-the-art facilities, modern buildings, and green spaces. It "
                                         "welcomes over 14,000 students from more than 100 countries. According to "
                                         "Times Higher Education World University Rankings, UNIC is ranked in the top "
                                         "501-600 universities globally, #47 for International Outlook, and #136 in "
                                         "the Eastern Europe and Central Asia region. UNIC also has subject-specific "
                                         "rankings in THE WUR, such as #151-175 for Business and Economics, "
                                         "#176-200 for Education, #251-300 for Psychology, #501-600 for Social "
                                         "Sciences, and #501-600 for Clinical and Health. Moreover, its faculty are "
                                         "internationally renowned and have extensive experience in scientific "
                                         "research."},
        {"role": "user", "content": "Data:\nUniversity of Nicosia\n-  Standard procedure to secure a position at the "
                                    "University\n  - Fill out an application online\n  - Provide high school living "
                                    "certificate or equivalency\n  - Provide proof of identity (international "
                                    "passport, two years valid)\n- Requirements\n  - 4 A-level certificates, "
                                    "5 ordinary level certificates with passing grades\n  - English proficiency "
                                    "certificate (IELTS or TOEFL)\n- Application rolling until October\n-  Extra "
                                    "payment of 40 euro for English test and 55 euro for admin fee\n- Contact for "
                                    "assistance via email or call\nUniversity of Nicosia\n-  Located in Republic of "
                                    "Cyprus\n- Offers Bachelor's in Business Administration and Law \n- Students must "
                                    "have a grade of C or higher in English and Mathematics to be admitted\n- "
                                    "Students cannot apply for two programs at the same time\n- Higher National "
                                    "Diploma can be used for credit transfer, but will need to be approved by "
                                    "Admissions Office\n- Students will not be able to use any exemptions from Higher "
                                    "National Diploma and will need to start from scratch\n-  Application Fee of 55 "
                                    "euros is required\n- English Proficiency Certificate is required for "
                                    "admission\n- All documents must be submitted by the end of the week for Spring "
                                    "semester admission \n- Admissions Office will respond the next day with a "
                                    "decision on admission based on the Higher National Diploma \n- If any questions, "
                                    "faculty can be contacted via email for further information\nUniversity of "
                                    "Nicosia\n-  Provides a campus program for bachelor's degrees \n- Requirements "
                                    "for admission include: \n    - Two forms to be filled in \n    - English "
                                    "proficiency certificate \n- Payment of 55 Euros for application fee \n- Need for "
                                    "letter of confirmation for English exam \n- Results of exam must be submitted "
                                    "before 10th of January for student visa\n\nUsing the data reply to the following "
                                    "message:\n\nMessage: What are the admission requirements for the University of Nicosia?"},
        {"role": "assistant", "content": "Of course. To apply to UNIC, you will need to fill out an "
                                         "online application form, provide proof of identity (international passport, "
                                         "two years valid) and provide your high school leaving certificate or "
                                         "equivalency. In addition, you will need to have 4 A-level certificates and "
                                         "5 ordinary level certificates with passing grades. An English proficiency "
                                         "certificate (IELTS or TOEFL) is also required. There is an extra payment of "
                                         "40 euro for the English test and 55 euro for the admin fee. If you have any "
                                         "questions or need assistance, please contact us via email or call us. Thank "
                                         "you!"},

    ]


def chat(df):
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_KEY")

    while True:
        message = input("User: ")
        if message == "quit":
            break
        messages_data.append({"role": "assistant", "content": message})
        message = f"Data:\n{''.join(search_takeaways(message, df))}\n\nUsing the data reply to the following message:\n\nMessage:{message}"
        messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.choices[0]['message'].content})
        print("UNICorn: ", response.choices[0]['message'].content)


if __name__ == '__main__':
    df = pd.read_json('data/nameless_embedding.jsonl', lines=True)
    df2 = pd.read_json('data/website_embedding.jsonl', lines=True)
    df3 = pd.read_json('data/conversation_embedding.jsonl', lines=True)
    df = merge_dataframes_3(df, df2, df3)
    chat(df)
