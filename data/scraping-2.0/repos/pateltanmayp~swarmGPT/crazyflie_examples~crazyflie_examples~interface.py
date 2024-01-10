import os
import openai
import langchain
import gradio as gr
import theme
from llm_helpers import Choreographer

#from waypoints_helpers import create_waypoints_data_structure, fly

# Find directory with cloned repo
for root, dirs, files in os.walk('/home/'):
    for name in dirs:
        if name == "swarmGPT":
            ROOT_DIR = os.path.abspath(os.path.join(root, name))

MUSIC_DIR = os.path.join(ROOT_DIR, "music")
CHORUS_DIR = os.path.join(ROOT_DIR, "chorus")
CONFIG_FILE = os.path.join(ROOT_DIR, "crazyflie/config/crazyflies.yaml")

openai.api_key = os.getenv("OPENAI_API_KEY")

#Find songs in MUSIC_DIR
song_list = [i[0:-4] for roots, dirs, files in os.walk(MUSIC_DIR) for i in files]

class Interface:
    def __init__(self,initialized, song_list):
        self.initialized = initialized
        self.data = []
        self.cg = Choreographer(music_dir=MUSIC_DIR, config_file=CONFIG_FILE, every_n_beats=2, x_lim=1, y_lim=1, z_lim=2)
        self.song_list = song_list
        print(self.cg.preset_prompt_templates.keys())

    def start(self, song_input):
        self.cg.set_song(song_input)
        beat_times = self.cg.get_beats(start_time=38, end_time=75)
        choreography, prompt = self.cg.choreograph(beat_times, "initial", custom_text="")
        data = self.cg.get_waypoints(choreography)

        self.data = data

        return data, prompt, gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)

    def prompt_type_selected(self, prompt_type_input):
        print(f"Prompt type {prompt_type_input}")
        if prompt_type_input == "custom":
            return gr.update(visible=True)
        return gr.update(visible=False)

    def follow_up(self, prompt_type_input, custom_text):
        if prompt_type_input not in self.cg.preset_prompt_templates.keys():
            return "No waypoints generated", "Please enter a valid prompt type!", gr.update(visible=False)
        
        if prompt_type_input == "custom" and custom_text == "":
            return "No waypoints generated", "Please enter a custom prompt!", gr.update(visible=False)

        custom_text = custom_text if prompt_type_input=="custom" else ""
        beat_times = self.cg.get_beats(start_time=38, end_time=75)
        
        choreography, prompt = self.cg.choreograph(beat_times, prompt_type_input, custom_text)
        data = self.cg.get_waypoints(choreography)

        self.data = data

        return data, prompt, gr.update(visible=False)
    
    def run(self):
        waypoints = create_waypoints_data_structure(self.data)
        fly(self.cg, waypoints)

        return "Hope you enjoyed the dance!", ""
    
    def make_invisible(self):
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
interface = Interface(False, song_list)

with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
    gr.Markdown("""# SwarmGPT""" ,elem_id="swarmgpt")
    gr.Markdown("""          Instructions:
    1. Enter song name and click on 'Start Choreographing'.
    2. Take a look at the outputted waypoints. If you're not satisfied, you can follow-up with the LLM using one of our preset prompt template or a custom one.
    Enter one of the following prompt types: redo, collision-avoidance, custom. If you selected the custom option, enter your prompt in the textbox that appeared.
    Then, click on follow-up.
    3. Once you're satisfied with the waypoints, click the "Let the Crazyflies Dance" button to start the dance!
    
    We hope you enjoy this demo :)""")
    with gr.Row():
        song_input = gr.Dropdown(choices=interface.song_list, label="Select Song")
        prompt_type_input = gr.Dropdown(choices=list(interface.cg.preset_prompt_templates.keys()), label="Enter Prompt Type:", visible = False, interactive=True)
    with gr.Row():
        custom_prompt = gr.Textbox(label="Enter Custom Prompt", visible=False)
    with gr.Row():
        prompt_output = gr.Textbox(label="Prompt", visible=True)
        ch_output = gr.Textbox(label="Waypoints", visible=True)
    with gr.Row():
        start_button = gr.Button("Start Choreographing")
        follow_up_button = gr.Button("Follow-up", visible=False)
    with gr.Row():
        run_button = gr.Button("Let the Crazyflies Dance", visible=False)
    
    
    #TODO: simulation button

    start_button.click(fn =  interface.start, inputs = [song_input], outputs = [ch_output, prompt_output, prompt_type_input, custom_prompt, follow_up_button, run_button]).then(
        lambda : gr.update(choices=list(interface.cg.preset_prompt_templates.keys())),
        inputs=None,
        outputs=prompt_type_input
    )
    #song_input.select(fn = lambda : gr.update(visible = False), inputs = None, outputs = [prompt_type_input, follow_up_button, run_button])
    song_input.select(interface.make_invisible, inputs = [], outputs = [prompt_type_input, follow_up_button, run_button])

    follow_up_button.click(fn = interface.follow_up, inputs=[prompt_type_input, custom_prompt], outputs=[ch_output, prompt_output, custom_prompt])
    prompt_type_input.select(fn = interface.prompt_type_selected, inputs=[prompt_type_input], outputs=[custom_prompt])
    custom_prompt.submit(fn = interface.follow_up, inputs=[prompt_type_input, custom_prompt], outputs=[ch_output, prompt_output, custom_prompt])
    #run_button.click(fn = interface.run, inputs=[], outputs=[ch_output, prompt_output])

if __name__ == '__main__':
    ui.launch()