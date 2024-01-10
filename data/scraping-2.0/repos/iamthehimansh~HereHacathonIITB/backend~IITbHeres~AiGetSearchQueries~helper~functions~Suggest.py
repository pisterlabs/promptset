from openai import OpenAI
import os,json
from .getLocals import getLocalShop
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")
file=open("/Users/iamthehimansh/Programing/heres_hacathon_iitb_team_5g_only/backend/story.txt","r")
SYSTEMPROMPT=file.read()
file.close()
USERPROMPTTEMPLATE=json.load(open("/Users/iamthehimansh/Programing/heres_hacathon_iitb_team_5g_only/backend/test.json","r"))


client = OpenAI(api_key=OPENAI_API_KEY)
# USERPROMPTTEMPLATE=settings.USERPROMPTTEMPLATE
# Give all search queries to the user for the given search query from open ai
def Suggession(userTask, lat, long)->str:
    shops=getLocalShop(lat,long)
    np=[]
    f={}
    # input("Press Enter to continue...")
    for i in shops["items"]:
        try:
            z={
                "title":i["title"],
                "category":i["categories"][0]["name"],
                "distance":i["distance"]
            }
            np.append(z)
        except:
            print(i)
            np.append(i)
    print(len(np),len(shops["items"]))
    f["nearbyPlaces"]=np
    f['tasks']=userTask
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo-16k",
    messages=[
        {"role": "system", "content": SYSTEMPROMPT},
        {"role": "user", "content": str(f)+"\n Return all posible task that can be done in the given location in an array or list"},
    ],
    max_tokens=4000,
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    print(Suggession("Pizza", 19.135032, 72.906146))