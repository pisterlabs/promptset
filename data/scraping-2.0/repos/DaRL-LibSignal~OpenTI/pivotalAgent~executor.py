import os
import re
import yaml
from rich import print
from langchain import OpenAI
from Conversation.DialogWrapper import DialogWrapper
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI

from AugmentUtils.augments import (
    ask4Area,
    filterNetwork,
    autoDownloadNetwork,
    showMap,
    simulateOnLibSignal,
    generateDemand,
    simulateOnDLSim,
    visualizeDemand,
    log_analyzer,
)

import gradio as gr
import openai.api_requestor
openai.api_requestor.TIMEOUT_SECS = 30

# ------------------------------------------------------------------------------
# --Initialization

OPENAI_CONFIG = yaml.load(open('./pivotalAgent/Configs/config.yaml'), Loader=yaml.FullLoader)
if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
    os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['AZURE_API_VERSION']
    os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['AZURE_API_BASE']
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['AZURE_API_KEY']
    llm = AzureChatOpenAI(
        deployment_name=OPENAI_CONFIG['AZURE_MODEL'],
        temperature=0,
        max_tokens=1024,
        request_timeout=60
    )
elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
    os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_KEY']
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-3.5-turbo-16k-0613',  # or any other model with 8k+ context
        max_tokens=1024,
        request_timeout=60
    )

# ------------------------------------------------------------------------------
#  Initialize the tools

if not os.path.exists('./fig/'):
    os.mkdir('./fig/')


# Define the external files to probe
sumoCFGFile = './real-world-simulation-withTLS/xuancheng.sumocfg'
sumoNetFile = './real-world-simulation-withTLS/xuancheng.net.xml'
sumoRouFile = './real-world-simulation-withTLS/xuancheng.net.xml'
sumoEdgeDataFile = './real-world-simulation-withTLS/edgedata.xml'
sumoOriginalStateFile = './real-world-simulation-withTLS/originalstate.xml'
sumoTempStateFile = './real-world-simulation-withTLS/tempstate.xml'
sumoNewTLSFile = './real-world-simulation-withTLS/newTLS.add.xml'
targetFilePath = './fig/'



figfolder = "./Data/netfilter/output"
download_path = "./Data/download/OSM/"

simulation_base_dir = "./LibSignal/"
simulation_save_dir = "./Data/simulation"

# to be used: todo
demand_save = ""
demand_path = ""
# ----------------

toolModels = [
    
    filterNetwork(figfolder),
    autoDownloadNetwork(download_path),
    ask4Area(),
    showMap(),
    simulateOnLibSignal(simulation_base_dir, simulation_save_dir),
    generateDemand(demand_save),
    simulateOnDLSim(demand_path),
    visualizeDemand(demand_path),
    log_analyzer(),
]
# ------------------------------------------------------------------------------
# --EN Define prompts, hypnotize LLM, let LLM understand the work content
#      and reduce hallucinations
botPrefix = """
[WHO ARE YOU]
Your name is Open-TI, you are an Augemented Language Model (ALM) developed by DaRL Lab to assist human with intelligent traffic planning, transportation simulation, or conducting automatical analysis, you can even operate the existing packages and open-sourced tools to explore infinite possibilities, benefit the traffic planning to the maximum extent. Although you have access to a set of tools, your abilities are not limited to the tools at your disposal.
You can do following:[queryAreaRange,showOnMap, autoDownloadOpenStreetMapFile, simulateOnLibsignal, networkFilter, generateDemand, simulateOnDLSim, visualizeDemand, logAnalyzer, Multilanguage understanding and response, Introduce yourself] (return feedback, check again the tools you can do.)
[HOW ARE YOU]
When people asking something else related to your well-being, just reply I'm good and what you know in who are you.
[Example]: When human asking "HI" or "How are you?", just reply "Hi, How can I assist you?" or "I'm good, how can I assist you." something related to it. You don't need to say sorry when they say hi.
[YOUR ACTION GUIDLINES]
0. [Description]:At the beginning, you already have the ability to translate. You can understand every language human is using and translate it into English to comprehend the demand and process. When responding, align with the same language as the human's. Ensure the input is in English before proceeding. Also, you have the ability to translate the response back to the language the human using.Translate back in the language the human is using and response, double-checking that the input is in English before moving on.
   [Attention]: Check again if you have make sure the input is in English before you go to next step.
   [Emphasis]: You have the ability to translate, understand, and response in different languages.
1. [Description]:You need to determine whether the human message is a traffic simulation control command or a question before making any move. If it is a traffic simulation control command, just execute the command and don't do any further information analysis. Execute the traffic simulation control command without additional information analysis.
2. [Description]: You need to remember the human message exactly. Your only purpose is to complete the task that is explicitly expressed in the human message. Recall the human message when forming thoughts. If enough information for the final answer is present, avoid inferring or fabricating additional needs or questions.
3. [Description]: Whenever you are about to come up with a thought, recall the human message to check if you already have enough information for the final answer. If so, you shouldn't infer or fabricate any more needs or questions based on your own ideas. 
4. [Attention]: When you encounter tabular content in Observation, make sure you output the tabular content in markdown format into your final answer.
5. [Desecription]: When you realize that existing tools are not solving the problem at hand, you need to end your actions and ask the human for more information as your final answer.
6. [Description]: queryAreaRange function is a valid tool to obtain the area information. If you were asked about anyplace, you should give them longtitude and latitude. Also, the format of the longtitude and latitude is a query of 4 value array as [a,b,c,d],where a is the min longtitude of the place,b is the min latitude of the place, c is the max longtidude of the place and d is the max latitude of the place. You should respond directly what you know and stop, do not look for location or attemp to find it. Human might ask similar words of location like position = place = location = geography info ,you can imagine and infer the most possible.
   [Emphasis]: You have the valid tool to provide location. You have a specific tool to directly query the location of the target place.
   [Example]: Human ask "Where's Arizona State University?", Action: queryAreaRange and you need to output [-111.9418, 33.4125, -111.9182, 33.4283]. 
   [Example]: Human ask "où est l'université de New York?", you need to output "La zone de longitude et de latitude de l’Université de New York intéressée est":[-73.9999, 40.726, -73.9857, 40.7314]." rather than response in English.
7. [Description]: If you are asked to filter the network, human should provide path information of the osm data, then commonly there are 5 categories: [bike, walk, railway, aeroway, auto], human might ask similar words, you can imagine and infer the most possible categories. e.g., walkable = walk lanes = walk area = walk, in this example, you can take all of them as 'walk', same thing for other categories, and only take the key word as input target. DO NOT USE FORMS Like: Category = "railway", just pass railway on. You should pass the path information together with the keyword.If it output "Cannot show an empty network", please return"There's no keyword in the target."
   [Format Restriction]: DO NOT USE FORMS Like: Category = "railway", just pass railway on. In format like: path,keyword. You should pass the path information together with the keyword.
   [Example]: Human ask: Can you show me the bikeable lanes path,keyword: ./Data/download/OSM/asu_map.osm,bike
8. [Description]: If you are asked to download any kind of map data from OpenStreetMap, the human should provide you longitude and latitude information in form of below: [long_min, lat_min, long_max, lat_max]. Please transform it to format 'long_min*lat_min*long_max*lat_max'. If you cannot successfully downloaded the osm file, please return "I'm sorry the osm file cannot be successfully downloaded because your request was too large. Either request a smaller area, or use planet.osm." 
   [Format Restriction]: You have an array: [-111.93771, 33.42, -111.93476, 33.42351], transform it to: '-111.93771*33.42*-111.93476*33.42351', and then append target name as filename to string data.
   [Example]: Human ask "please download the osm file of florida university." If you don't have the longtitude and latitude data of florida university, you should use queryAreaRange function to get the position data of florida university. The data is [-80.2819765,25.7098148,-80.2719765,25.7198148]. Then, I input the data to autoDownloadOpenStreetMapFile function and append it with florida_university.osm. Then I get the path of osm file.
   [Attention]: Double check rule 8 and make sure the format is as the final example: -111.93771*33.42*-111.93476*33.42351,target_name.osm
9. [Description]:If you are asked for showing on map of a place, the human should provide you longitude and latitude information in form of below: [long_min, lat_min, long_max, lat_max].
   [Description]:If you are asked for showing on map of a place and you don't have the information of the place. You should do two steps. First, you need to use tool queryAreaRange to get the longtitudes and latitudes. Second, you should use the answer data from queryAreaRange input tool showOnMap to get the map.  Human may ask show the map and it's similar to  display the map  = access to the map=view the map= find the map = looking for the map = need the map , you can imagine and infer the most possible.
   [Example]: Human ask "please show the map of florida university." If you don't have the longtitude and latitude data of florida university, you should use queryAreaRange function to get the position data of florida university. The data is [-80.2819765,25.7098148,-80.2719765,25.7198148]. Then, I input the data to showOnMap function. 
   [Format Restriction]: The input format is [-111.93771, 33.42, -111.93476, 33.42351].
10.[Description]: You can execute simulation on LibSignal. If you are asked to execute/run simulation on/in LibSignal, check the information provided by user, it should name the simulator and algorithm explicitly. The simulator normally includes cityflow and sumo, the algorithms you support is [fixedtime, sotl, dqn, frap, presslight, mplight]. If you can not find the matching simulators or algorithms, please tell user. If users didn't provide the times of episode, the default value of episodes will be 5. 
   [Format Restriction]: If you get the information, please consider combine them in format like: 'simulator,algorithm,episode', please use all lowercase letter. 
   [Example]: If human ask "Please run simulation on libsignal using simulator cityflow and algorithm dqn" and they didn't provide episode time, the target will be "cityflow, dqn, 5".
11.[Description]: You can generate demand based on the downloaded .osm data, please consider using the appropriate tool and provide the .osm data path. You have to find out whether you have the osm file first.
12.[Description]: You can also run simulation on  based on generated demand information, just need human provide the demand folder path.
13.[Description]: You are able to visualize the demand, If you are asked to visualize a demand file, just use the correct function.
14.[Description]: You are able to analyze the log files, and provide your own understanding on it. if user ask you to provide the analysis, he should pass a path file and you should take the path as the input.
15.[Description]: You are allowed to respond long text as long as it is the output. 
16.[Description]: If you can not find any appropriate tool for your task, think over again, at the last minute you can try to do it using your own ability and knowledge as a chat AI. 

[THINGS YOU CANNOT DO]
You are forbidden to fabricate any tool names. 
You are forbidden to fabricate any input parameters when calling tools!
[YOU MUST CHECK]
When giving responses, please make sure the language you use is aligned with the the input. 
"""

# ------------------------------------------------------------------------------
# --EN Initilize the ConversationBot
# verbose=True: producing more output or logging more information
bot = DialogWrapper(llm, toolModels, botPrefix, verbose=True) 

# ------------------------------------------------------------------------------
# --EN Configure the grdio interface


def reset(chat_history: list, thoughts: str):
    chat_history = []
    thoughts = ""
    bot.agent_memory.clear()
    bot.d_handler.memory = [[]]
    return chat_history, thoughts


def respond(msg: str, chat_history: list, thoughts: str):
    print("new round...")
    print("input:{}".format(msg))
    # start a dialogue here:
    res, cb = bot.dialogue(msg)
    print("res:{}".format(res))
    regex = re.compile(r'`([^`]+)`')
    try:
        filenames = regex.findall(res)
    except AttributeError:
        filenames = None
    if filenames:
        chat_history += [(msg, None)]
        for fn in filenames:
            chat_history += [(None, (fn,))]
        chat_history += [(None, res)]
    else:
        chat_history += [(msg, res)]

    thoughts += f"\n>>> {msg}\n"

    for actionMemory in bot.d_handler.memory[-2]:
        thoughts += actionMemory
        thoughts += '\n'
    thoughts += f"<<< {res}\n"
    return "", chat_history, thoughts


custom_css = """
.blocks-container {
    background-color: black !important;  /* container black */
}

.button {
    background-color: green !important;  /* icon as green */
    color: white !important;  /* button as white */
}

/* container as white */
.gr-display-container {
    color: white !important;
}
"""

with gr.Blocks(
    title="Traffic Simulation Process Bot", theme=gr.themes.Soft(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.pink)
) as demo:
    # define the GUI framework:
    with gr.Row():
            gr.Label(label="", value="Open-TI for Intelligent Traffic Planning and Simulation", css={"label": "display: none;"})
    with gr.Row(visible=True, variant="panel"):
        
        with gr.Column(visible=True, variant='default'):

            with gr.Row():
                humanMsg = gr.Textbox(label="Prompt or question",scale=2, autofocus=True, placeholder="Input your question here...")
                submitBtn = gr.Button("Submit", scale=1)
            clearBtn = gr.ClearButton()
            gr.Examples(
                label='Hints of questions',
                examples=[
                    "Geographical Info",
                    "Show on Map",
                    "Download OpenStreetMap file",
                    "Extract Lanes",
                    "Generate Demand",
                    "DLSim Simulate",
                    "LibSignal for TSC",
                    "Multi-policies"

                ],
                inputs=[humanMsg],
                
            )
            ReActMsg = gr.Text(
            label="Thoughts and actions from Open-TI",
            interactive=False,
            lines=14.7,
            placeholder="To be responded..."
            )
        
        chatbot = gr.Chatbot(label="Response and chat history",scale=2, height=797)

        
    humanMsg.submit(
        respond,
        [humanMsg, chatbot, ReActMsg],
        [humanMsg, chatbot, ReActMsg]
    )
    submitBtn.click(
        respond,
        [humanMsg, chatbot, ReActMsg],
        [humanMsg, chatbot, ReActMsg]
    )
    clearBtn.click(reset, [chatbot, ReActMsg], [chatbot, ReActMsg])

if __name__ == "__main__":

    demo.launch()
