import os
# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
from langchain_core.messages import HumanMessage
# import icecream as ic

from PIL import Image

from image_depth import get_depth_map_image,load_midas_model,load_transformations

# from trulens_eval.feedback import Feedback

from dotenv import load_dotenv
load_dotenv()

# genai.configure(api_key=GOOGLE_AI_STUDIO) ## if we do not use langchain
# os.environ["GOOGLE_API_KEY"] = GOOGLE_AI_STUDIO ## when using the langchain

def get_prompt_templates():
    intent_prompt_template = """Your job is to identify intent of the question. There are only two inents ['Photo is required','No photo required']
    Description for each intent is given below.
    -Photo is required: If the user wants to know what is happening around him, who is near him or any object closer to him. Further, user might ask about places such as food stalls, shops, bus halts, etc.
    -No photo required: if the user does not ask about what is happening around him, who is near him or any object closer to him.

    Please return the specified intent as the answer at the end.
    
    Qestion: {question}
    Answer:
    """

    main_chat_system_prompt_template = """You are an assistant to a blind person. If the context is provided please use the provided \
        context and answer accurately and truthfully based on the context. Use the given current conversation history appropriately if required.
        If you are not sure about your answer please tell that 'I am not sure'.

        Current Conversation: {memory}

        Context: {context}
        
        User: {query}
        Assistant:  
        """
    

    intent_prompt = ChatPromptTemplate.from_template(intent_prompt_template)
    main_chat_system_prompt = ChatPromptTemplate.from_template(main_chat_system_prompt_template)
    return intent_prompt,main_chat_system_prompt

def main():

    llm_chat = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.1,
        top_k=3,
        top_p=0.7,
    )

    llm_vision = ChatGoogleGenerativeAI(
        model="gemini-pro-vision"
    )

    GENERAL_VISION_PROMPT = """You are and expert in analysing the images and you need to provide detailed explanation \
        of the provided images. You will be provided RGB image and a depth map image for the same RGB image. Your detailed explanation must include \
        all the identifiable objects and approximated distance to each object. When providing the approximate distance please use the assistance of provide depth image map appropriately. Do not make up answers. \
        If you make up answers you will be fined $2000 and each accurate description you will be rewarded $100. \
        In the depth image map, the brighter areas are closer than the darker areas. \
        Further, try to predict the nature of the environment \
        using the provided RGB image and provide possible climate details as well. Apart from that, try to find relevant details for the below \
        given asked question as well and answer appropriately. Your should imitate the tone of a human when answering the questions and instead of image you should use the word 'view'. If any text is in languages other than English please translate them. Question: {query}
        Answer: """


    intent_prompt,main_chat_system_prompt = get_prompt_templates()

    intent_analyser_chain = RunnableMap(
        {
            "question": lambda x: x['question'],
        }
    ) | intent_prompt | llm_chat | StrOutputParser()


    ## Todo: 
    chat_bot_chain = RunnableMap(
        {
            "query": lambda x: x['query'],
            "context": lambda x: x['context'],
            "memory": lambda x: x['memory']
        }
    ) | main_chat_system_prompt | llm_chat | StrOutputParser()



    ## midas model
    midas_model = load_midas_model(model_type="DPT_Hybrid")
    midas_transforms = load_transformations()


    chat_history = []
    while True:

        user_input = input("User ('q' to exit): ")

        if user_input == "q":
            break

        intent = intent_analyser_chain.invoke({"question" : user_input})
        print(intent)

        final_vision_prompt = GENERAL_VISION_PROMPT.format(query=user_input)

        vision_response = ""
        if intent == "Photo is required":

            rgb_image = Image.open("images/london-street-view-songquan-deng.jpg")
            rgb_image = rgb_image.convert("RGB")

            depth_image = get_depth_map_image(image=rgb_image,midas_model=midas_model,midas_tranforms=midas_transforms)
            # print(depth_image.size,rgb_image.size)
            
            tmp_message = HumanMessage(
                content=[
                    {
                        "type":"text",
                        "text": final_vision_prompt
                    },
                    {
                        "type":"image_url",
                        "image_url": rgb_image
                    },
                    {
                        "type":"image_url",
                        "image_url": depth_image
                    }
                ]
            )

            vision_response = llm_vision.invoke([tmp_message]).content
            # print(vision_response.content)
        
        chat_bot_response = chat_bot_chain.invoke(
                        {
                            "query": user_input,
                            "context": vision_response,
                            "memory": chat_history
                        }
                    )

        #update the history
        chat_history.append({"User: " : user_input})
        chat_history.append({"Assistant: " : chat_bot_response})

        print("Chatbot: ",chat_bot_response)
        print()


if __name__=="__main__":
    main()