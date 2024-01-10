import os
from typing import Any, Dict, List, Optional
from pydantic import Extra
from langchain.vectorstores import DeepLake
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, SequentialChain, SimpleSequentialChain, ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.openai import OpenAIChat
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.memory import ConversationBufferMemory
import openai
import librosa
import numpy as np
import yaml
from yaml.loader import SafeLoader

openai.api_key = os.getenv("OPENAI_API_KEY")

class MusicChain(LLMChain):
    """
    A custom chain for music processing.
    """

    prompt: PromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  # :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)
        print(f"\nFull prompt: {prompt_value}")

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(f"Prompt: {prompt_value}")
            run_manager.on_text(f"Response: {response}")
        
        

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"
    
class Choreographer():

    def __init__(self, music_dir, config_file, every_n_beats, x_lim, y_lim, z_lim):
        self.music_dir = music_dir
        self.config_file = config_file
        self.every_n_beats = every_n_beats # Change formation every __ beats
        self.limits = {'x': x_lim, 'y': y_lim, 'z': z_lim} # Defines boundaries of permissible flying area
        self.agents = {}
        self.starting_pos = {}
        self.num_drones = 0
        self.song = ""
        self.artist = ""
        self.preset_prompt_templates = {}
        self.all_waypoints = np.zeros((2, 6)) # For concatenated array shape
    
        self.configure_drones()

    def configure_drones(self):
        # Open the file and load the file
        with open(self.config_file) as f:
            data = yaml.load(f, Loader=SafeLoader)
            robots = data["robots"].values()
            
            for i, robot in enumerate(robots):
                self.agents[i] = int(robot["uri"][-1])
                self.starting_pos[i] = list(robot['initial_position'])
        
        self.num_drones = len(self.agents.values()) # Number of drones
        print(f"\nNumber of drones detected: {self.num_drones}")

    def set_song(self, song: str):
        self.song = song
        #self.artist = artist

        # Reset LLM state
        self.all_waypoints = np.zeros((2, 6)) # For concatenated array shape
        self.preset_prompts = {}
        self.configure_llm(song) # Keep drone info same between songs, just update the LLM setup

    def configure_llm(self,song):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        print(openai.api_key)

        llm = OpenAIChat(max_tokens=10000, model='gpt-3.5-turbo-16k')

        structure = PromptTemplate(input_variables=["chat_history", "prompt"], template="""You're an assistant that will help a human choreograph a dance for a set of small drones.

{chat_history}
Human: {prompt}
AI: """)
        
        # Make the chain available to whole class, we want it to persist until a new song is set
        self.chain = MusicChain(
            prompt=structure,
            llm=llm,
            memory=ConversationBufferMemory(memory_key="chat_history")
        )

        self.preset_prompt_templates["initial"] = PromptTemplate(input_variables=["song", "number", "beat_times", "starting_pos", "format_instructions"],
            template="""Choreograph a harmonized, symmetric dance for {number} drone that reflects the mood of the song {song}. \
The permissible flying region is a cube with a side length of 2 metres, centred at the origin. {starting_pos} \
Format your output as a series of waypoints for each drone, one at every beat in the song. The beat times are {beat_times}. \
Make sure the waypoints will not lead to any collisions between drones. Drones must not arrive at the same waypoints simultaneously, otherwise they will collide. \
Also, when transitioning between waypoints, drones must not cross the same point (this would also create a collision). \
The drones should move gently and artistically, creating sophisticated formations together. \
Make sure the drones don't touch the ground (their z coordinate should always be greater than 0). \
    {format_instructions}""")
        
        self.preset_prompt_templates["collision-avoidance"] = PromptTemplate(input_variables=[], template="Will the waypoints you've created result in any collisions between drones? If so, please modify them until there won't be collisions. Also, drones shouldn't fly directly over each in close proximity. Follow the exact same format instructions as your last output.")
        self.preset_prompt_templates["redo"] = PromptTemplate(input_variables=[], template="The waypoints you generated will create a mundane or boring dance. Could you make it more artistic and creative, while ensuring there won't be collisions? Follow the exact same format instructions as your last output.")
        self.preset_prompt_templates["custom"] = PromptTemplate(input_variables=["text"], template="{text} Follow the exact same format instructions as your last output.")
        
    def get_beats(self, start_time: int, end_time: int) -> list:
        filename = os.path.join(self.music_dir, f"{self.song.lower()}.mp3")

        y, sr = librosa.load(filename, offset=start_time, duration=(end_time-start_time)) #y = waveform, sr = sampling rate

        y_harmonic, y = librosa.effects.hpss(y) # Get percussive waveform and place back into y
        tempo, beat_times = librosa.beat.beat_track(y = y, sr = sr, units="time")

        print("\nEstimated tempo: {:.2f} beats per minute".format(tempo))

        # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print("\nBeat times: ", beat_times)

        # Due to token limits, truncate the waypoints after certain point
        beat_times = beat_times[self.every_n_beats::self.every_n_beats]
        # beat_times = beat_times[:8*self.every_n_beats:self.every_n_beats]

        return beat_times.tolist()
        
    def choreograph(self, beat_times: list, prompt_type: str, custom_text: str) -> dict:
        
        # Initial positions were retrieved from crazyflies.yaml
        starting_pos = ""
        for i, pos in self.starting_pos.items():
            starting_pos += f"Drone {i} has an initial position of {pos}, "
        
         # Specify output format
        response_schemas = [ResponseSchema(name=f"Waypoints for drone {i}", description=f"The waypoints corresponding to drone number {i}") for i in range(self.num_drones)]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        format_instructions += " All waypoints should be in the following format, including both the open and close brackets: [time in seconds, x coordinate in metres, y coordinate, z coordinate, yaw angle in degrees]. Waypoints for a particular drone should be enclosed in brackets and separated by a semi-colon. The following is an example. 'Waypoints for drone 1': '[3.99, 0.5, 0.5, 0.5, 0]; [5.76, 0.4, -0.6, 0.4, 180]; [7.49, -0.2, 0.1, 0.6, 90]', ... Ensure the json entries (one for each drone) are separated by a comma."

        prompt_template = self.preset_prompt_templates[prompt_type]

        # Only initial prompt will contain drone data, etc - not needed in future prompts
        if prompt_type == "initial":
            prompt = prompt_template.format(song=self.song, number=self.num_drones, beat_times=beat_times, starting_pos=starting_pos, format_instructions=format_instructions)
        elif prompt_type == "custom":
            prompt = prompt_template.format(text=custom_text)
        else:
            prompt = prompt_template.format()

        print(f"\nPrompt: {prompt}")
        result = self.chain.predict(prompt=prompt)
        print(f"\nResult: {result}")

        output = output_parser.parse(result)
        return output, prompt
        
    def get_waypoints(self, llm_output: dict) -> np.ndarray:

        # Waypoint format: [cfid, time, x, y, z, yaw]
        for i, waypoint_set in enumerate(llm_output.values()):
            waypoint_set = waypoint_set.replace('(','').replace(')','').replace('[','').replace(']','').replace(' ', '')
            waypoints = [waypoint.split(',') for waypoint in waypoint_set.split(';')]
            waypoints = np.array(waypoints).astype(float)

            agent_col = np.zeros((waypoints.shape[0], 1)).astype(int) + self.agents[i]
            waypoints = np.concatenate((agent_col, waypoints), axis=1)
            self.all_waypoints = np.concatenate((self.all_waypoints, waypoints), axis=0)

        self.all_waypoints[self.all_waypoints[:, 0].argsort()] # Sort by agent ID

        # Remove initial 2 rows (were for shape only)
        self.all_waypoints = self.all_waypoints[2:, :]

        # Ensure all waypoints are within limits
        for i, key in enumerate(self.limits.keys()):
            self.all_waypoints[:, i+2] /= (np.abs(self.all_waypoints[:, i+2]).max() / self.limits[key]) if self.all_waypoints[:, i+2].max() != 0 and self.all_waypoints[:, i+2].any() > self.limits[key] else 1

        return self.all_waypoints

def main():
    pass

if __name__ == "__main__":
    main()