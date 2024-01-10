import os
from dotenv import load_dotenv
load_dotenv()


from langchain.llms import Clarifai
from langchain import PromptTemplate
key = os.environ.get('key')
print(key)
llm = Clarifai(pat=key, user_id='meta', app_id='Llama-2', model_id='llama2-7b-chat')


def chatbot(context,human_question):
    try:
        

        template = """Answer the question based on the context below. If the
        question cannot be answered using the information provided answer
        with "Sorry I don't know". answer shouldn't exceed more than 50 words.

        Context: {context}

        Question: {human_question}

        """

        prompt_template = PromptTemplate(
            input_variables=["context","human_question"],
            template=template
        )

        
        return True,llm(
    prompt_template.format(
        context=context,human_question=human_question
        )
    )
        
        
    except Exception as e:
        print(e)
        return False,None
# chatbot("What makes physics so cool? To me, physics is awesome and interesting because it asks the one question no other [physical] science does: “how?” Physics looks at how things happen-the reason things happen-and explains it. No other [physical] science does this. Chemistry deals with the result of things happening while biology deals with why things can happen, but physics deals with the how things happen. Take, for instance, a moving car. A biologist would look at a moving car and wonder \"why is it able to move?\" He would probably ask himself, “What is it about the way those parts work together that make it go forward? Why is it able to happen?” A physicist, on the other hand, sees a moving car and wonders \"how can the car move?\" \"What is providing a forward motion? How are the wheels even able to turn?\" And, on the other end of the spectrum, we have a chemist who... would probably see a moving car and wonder what would happen if he threw sulfuric acid in the tank. But what is physics? Well, the ultimate goal of physics is to understand the universe around us. This ranges from tiny, inter-molecular particles, to huge things like entire solar systems. Physics argues that there must exist fundamental rules that govern the how of everything around us, and it is physicists’ job to find out what those rules are and apply formulas to them. It is really is fascinating how much we already know, yet mindboggling how much we don’t. Such simple mechanics, such as Newton’s famous “force equals mass times acceleration” equation can help scientists understand the basic motion of nearly anything. So, I’m here to teach you guys these principles, too. That answers what the basic goal of physics is, but doesn’t answer what we actually do in physics. Well, in its simplest form, physics is “applied mathematics.” Now, you’ve probably heard the term “applied blank” before, but I’m not sure you know what that really means. To answer that, think about a toolbox. Essentially, all of the math you’ve learned since elementary school are tools that were put into this toolbox. We all start off with an empty toolbox, and as we get older and learn more complicated math, we sort of “buy and learn how to use” new tools. By the time we hit physics, we have a good set of tools that we’re very familiar with using. But, we never really used them before. See, mathematics is the class that teaches you what the tools are and refines your ability to use them. But, you never truly use them. So, what’s the purpose of knowing these math concepts? Well, what’s the purpose of owning and learning how to use tools? To build things! In physics, we take the tools that we’ve learned in our math classes and apply them to real world problems. For example, if we want to find out how high a ball that was thrown upwards will travel, we can use the algebra that we learned nearly two years ago so figure it out. All physics really is is just applying, or should I say finally using, all of the tools you’ve accumulated throughout your education to figure out the way the world behaves. Now, physics has a bad reputation as being one of the hardest sciences you can take in high school. This can be true, but I believe every concept can be broken down and simplified. That’s why I’m here to teach you guys the wonders of physics! From the fundamental Newtonian equations to the famous e=mc^2, together we will embark on a journey to better understand the world around us. So, take a step back, and prepare to see the awesomeness that is the world of physics.","defination of physics?")