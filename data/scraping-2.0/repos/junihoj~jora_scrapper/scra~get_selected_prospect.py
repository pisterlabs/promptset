import openai
import re

API_KEY = open('api_key.txt', 'r').read()
openai.api_key = API_KEY

def get_employee_ids(list_of_prospect):
    list_of_ids = []
    for staff in  list_of_prospect:
        list_of_ids.append(staff['id'])
    return list_of_ids


def get_right_prospect(list_of_staff):
    employee_job_titles = [employee_data['title'] for employee_data in list_of_staff if employee_data['title'] is not None]
    comma_separated = "\n".join(employee_job_titles)
    # print("COMMA SEPARATED", comma_separated)

    gpt_promt = """here are list of job title of people in a company {} I want you to give me the top three of these job title in order of taking decisions in the company I want you to give your answer in this format exactly like
    1.first\n2.second \n3.third 
    """.format(comma_separated)

    # no list or numbering just comma separated job titles
    chat_log = [{"role":"user", "content":gpt_promt}]
    initial_response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=chat_log
        )

    assistant_response = initial_response.choices[0].message.content

    # print("ASSISTANT ANSWER", assistant_response)
    # print("GPT RESPONSE", re.split( r'(\d.\s)', assistant_response))
    splitted_response = re.split( r'(\d.\s)', assistant_response)
    # matched  = ['Co President of Arise', 'President, Global Business Solutions']
    matched = [title.replace("\n","") for title in splitted_response if title.replace("\n","").strip() in employee_job_titles ]
    selected_prospect = [k for k in list_of_staff if k['title'] in matched]
    employees_ids = get_employee_ids(selected_prospect)
    print(employees_ids)
    return employees_ids
    # return selected_prospect



sample_prospect = [{"id": "366b0acc16f4466ec3a57b24a5f80a6094f3fd73f2c3db153c039cc77322dbdc", "avatar": None, "name": {"fullName": "Ruoxing Sun"}, "title": "Senior Software Development Engineer, Tech Lead, Tech Owner"}, {"id": "8ec0e93f0039e2ee80fbc732e5b8d266aa66a6e8f7700d1a54c25b625963abb7", "avatar": None, "name": {"fullName": "Angela Torres"}, "title": "Team Lead, Brand Partnerships"}, {"id": "bc3ad29a33a208d5f8a4702fd9ae97b59c897ebcf00513f4e7b565eb862a11c6", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/bc3ad29a33a208d5f8a4702fd9ae97b59c897ebcf00513f4e7b565eb862a11c6", "name": {"fullName": "Christine Ruecker"}, "title": "Co President of Arise"}, {"id": "ea8f2672ce4da6f51cbc0808d268be0c03c342973219ba0631ec57da678ec1d4", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/ea8f2672ce4da6f51cbc0808d268be0c03c342973219ba0631ec57da678ec1d4", "name": {"fullName": "Blake Chandlee"}, "title": "President, Global Business Solutions"}, {"id": "66bbdc8689f769ab0ca0eeef9e8fe17db4de567e638a3953888b61283dcd5c94", "avatar": None, "name": {"fullName": "James Liu"}, "title": "Head of HR, Lifecycle Operations, Americas"}, {"id": "26009181309e51ff3d7393f15d199a1c0d22ded26f3017b651710617da5e0bdc", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/26009181309e51ff3d7393f15d199a1c0d22ded26f3017b651710617da5e0bdc", "name": {"fullName": "Thomas Grainger"}, "title": "Creator Engagement Manager, Pride Tiktok President"}, {"id": "0797509148a70bc3c46fcb41ceba78c38dc5d41f265d089f7e364b69b173df89", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/0797509148a70bc3c46fcb41ceba78c38dc5d41f265d089f7e364b69b173df89", "name": {"fullName": "Alexa Scordato"}, "title": "Chief of Staff"}, {"id": "7878d0c5d0727893db7cc3c21cff5bcc9b251c448c1a91bee2f2380189c01bb3", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/7878d0c5d0727893db7cc3c21cff5bcc9b251c448c1a91bee2f2380189c01bb3", "name": {"fullName": "Hilary McQuaide"}, "title": "VP, Global Communications"}, {"id": "acc0d7bbee5afa78eba9b99e3c26c0e51d266010cdcb86532aaecad2e9d09efc", "avatar": None, "name": {"fullName": "Patrick Nommensen"}, "title": "GM of UK E-commerce"}, {"id": "ce012f89998a6943651746302dc2fade6efd441bca26aede4e0e0db61af9bfbb", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/ce012f89998a6943651746302dc2fade6efd441bca26aede4e0e0db61af9bfbb", "name": {"fullName": "Andrea Herskowich"}, "title": "Industry Relations Manager, Global Business Marketing"},
                   {"id": "47b3bcf9df790ddd60267f08c01845e49233fc2660c1e93456c0d5416c52a810", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/47b3bcf9df790ddd60267f08c01845e49233fc2660c1e93456c0d5416c52a810", "name": {"fullName": "Brett Armstrong"}, "title": "Co-gm, Australia"}, {"id": "548329b5394dad7eb9fc9163d5b0483a518052b18f7e90f13bed92f69f96c6be", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/548329b5394dad7eb9fc9163d5b0483a518052b18f7e90f13bed92f69f96c6be", "name": {"fullName": "Angie Wright"}, "title": "Client Partner"}, {"id": "bd12e2dc25df2b43848390029cfc8ff467880a63941a3608323c3eacf8b0d1ca", "avatar": None, "name": {"fullName": "Alex Barreto"}, "title": "Chief Compliance Officer Brazil, Global Payment-compliance Team"}, {"id": "ebcdad274250206763fab59320b000abbe22e7a50238076caf9c3ac3ffd80b6d", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/ebcdad274250206763fab59320b000abbe22e7a50238076caf9c3ac3ffd80b6d", "name": {"fullName": "Gabriela Chaves Schwery Comazzetto"}, "title": "GM Global Business Solutions Latam, Brazil Tiktok, Bytedance"}, {"id": "1edb9112cc96fa7045460967914f5d395cd93604c1f6ab727e52a7c94125942d", "avatar": None, "name": {"fullName": "Kinda Ibrahim"}, "title": None}, {"id": "51c7f422475e93f000558352254540bb8f29f4899704ff648012841f900e39e5", "avatar": None, "name": {"fullName": "Philip Packer"}, "title": "Joint CEO"}, {"id": "7c34572cd6b330a324a3ef87cf2533088ac0bc6e74b2e73dabee72ed8bce12e0", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/7c34572cd6b330a324a3ef87cf2533088ac0bc6e74b2e73dabee72ed8bce12e0", "name": {"fullName": "Richard Waterworth"}, "title": "GM, Europe"}, {"id": "b3d21bd43130f5730dc6fbb81f80888fab4ceb89698102437756206cd6082e8b", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/b3d21bd43130f5730dc6fbb81f80888fab4ceb89698102437756206cd6082e8b", "name": {"fullName": "Stuart Flint"}, "title": "VP, Europe, Sales"}, {"id": "1323152a6d94c82a87304a071192cb2e49e9de8134ca1a1132fcd3d27cf31563", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/1323152a6d94c82a87304a071192cb2e49e9de8134ca1a1132fcd3d27cf31563", "name": {"fullName": "Zenia Mucha"}, "title": "Chief Brand, Communications Officer"}, {"id": "feda4f3b3a273d5bd0af40671ed447d75d8a7a0e18ba30f4ab728a877205ec20", "avatar": "https://d2pbji6xra510x.cloudfront.net/v2/avatars/feda4f3b3a273d5bd0af40671ed447d75d8a7a0e18ba30f4ab728a877205ec20", "name": {"fullName": "Allen Licup"}, "title": "Chief Compliance Officer"}]

selected = get_right_prospect(sample_prospect)
