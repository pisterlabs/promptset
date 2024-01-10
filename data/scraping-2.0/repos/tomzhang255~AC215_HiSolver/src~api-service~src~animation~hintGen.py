import os
import json

from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from math import sqrt, cos, sin
from typing import Union
from typing import Optional
from manim import *
import random
import promptlayer
promptlayer.api_key = ""
from langchain.callbacks import PromptLayerCallbackHandler
from langchain.prompts import ChatPromptTemplate


def predict_custom_trained_model_sample(project, endpoint_id, location, instances):
    # Import the necessary libraries
    from google.cloud import aiplatform

    # Initialize the client
    client = aiplatform.gapic.PredictionServiceClient()

    # Prepare the endpoint
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )

    # Prepare the payload
    # Ensure that instances is a list of dictionaries
    payload = {"instances": [instances]}

    # Make the prediction request
    response = client.predict(name=endpoint, payload=payload)

    return response

def animation_from_question(video_name, query):
    

    prompt_template = '''You are a math function (eg. 5x^2 + 5) extractor tool. From the below question, come up with the most meaningful function.
    
    "{query}" 
    Most meaningful FUNCTION (with x as variable) is: '''

    try:
        hint_function = predict_custom_trained_model_sample(
            project="113408214330",
            endpoint_id="3290556826956857344",
            location="us-east4",
            instances={"prompt": query} # Update with your actual data
        )
    except Exception as e:
        print("Error in custom model prediction:", e)
        llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )
        hint_function = llm_chain(query)
        

    def GraphAgent(query = ""):
            

        llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0)

        tools = [FunctionGraphTool()]

        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, handle_parsing_errors=True)
        agent(query)
        return


    class FunctionGraphTool(BaseTool):
        name = "Graph_Function"
        description = (
        "use this tool when you need to graph a function"
        "To use the tool, you must provide the python lambda function as a string. Give the function in python syntax (ex. lambda x: 2*x**5 - 5*x**2 + 1)"
        "['function']."
    )

        
        def _run(self, 
                function):
            print(function)
            lambda_f = eval(function)
            _, expression = function.split(":", 1)
            # Form the desired equation string
            function = "y = " + expression.strip()
            x_range, y_range = determine_range(lambda_f)
            x_step = determine_step_size(x_range)
            y_step = determine_step_size(y_range)
            x_range = (*x_range, x_step)
            y_range = (*y_range, y_step)  
            print(x_range, y_range)

            return create_graph(lambda_f,function,x_range,y_range)

        def _arun(self, radius: int):
            raise NotImplementedError("This tool does not support async")


    def create_graph(lambda_f,function,x_range,y_range):
        # Create an instance of the scene and set the circle radius
        scene = HotelCostGraph(lambda_f,function,x_range,y_range)
        scene.render()

        video_path = f"./media/videos/1080p60/"
        os.rename(video_path + "HotelCostGraph.mp4", video_path + video_name)
        
        return "graph generated successfully, check your filesystem"


    class HotelCostGraph(Scene):
        def __init__(self, lambda_f, function, x_range, y_range):
            super().__init__()
            self.function = function
            self.lambda_f = lambda_f
            self.x_range = x_range
            self.y_range = y_range

        def construct(self):
            # Add 1 to the second element
            temp_list = list(self.y_range)
            temp_list[1] += .1
            temp_list[0] -= .1
            self.y_range = tuple(temp_list)
            continuousAxes = Axes(
                x_range=self.x_range,
                x_length=5,
                color=BLUE,
                y_range=self.y_range,
                y_length=4,
                axis_config={
                    "tip_width": 0.15,
                    "tip_height": 0.15,
                    "include_numbers": True,
                    "font_size": 14,
                })


            # Create the first text separately due to different positioning and font size
            intro = Tex(r"\parbox{10cm}{" + self.function + "}", font_size=20).move_to(UP * 3 + LEFT * 3.5)
            self.play(Create(intro))
            self.wait(.1)


            self.play(Create(continuousAxes))


            bright_colors = [YELLOW, RED, GREEN, BLUE, PINK, PURPLE, ORANGE, WHITE]
            selected_color = random.choice(bright_colors)

            continuousFunc = continuousAxes.plot(self.lambda_f, color=selected_color)
            graph_label = continuousAxes.get_graph_label(continuousFunc, label=self.function).scale(.6).shift(LEFT * .8).shift(UP * 1.1)
            self.wait(.1)
            self.play(Create(continuousFunc), run_time=2)
            self.play(Write(graph_label))

            self.wait(.1)
            return 0

    
    GraphAgent("draw me the fuction lambda x:" + hint_function['text'])

    return 0



################################## Utility functions #################################

import numpy as np
from scipy.signal import find_peaks

import re

def split_text_to_equations(text):
    
    text = text.replace('$', '\\$')
    text = text.replace('%', '\\%')
    
    lines = text.split('\n')
    result = []

    inside_equation = False
    temp_equation = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if inside_equation:
            temp_equation += line + "\n"
            if r"\]" in line:
                result.append(temp_equation.strip())
                temp_equation = ""
                inside_equation = False
        else:
            if r"\[" in line:
                inside_equation = True
                temp_equation += line + "\n"
            else:
                result.append(line)

    return result



import numpy as np

def determine_range(f, x_start=-100, x_end=100, samples=100, focus_on_y_intercept=True):
    """
    Determines a tight suitable range for plotting the lambda function, focusing on areas of high curvature or the y-intercept.
    
    Args:
        f (func): Lambda function.
        x_start, x_end (int, int): Initial range for x to sample.
        samples (int): Number of samples to take within the range.
        focus_on_y_intercept (bool): If True, focuses on y-intercept for simple lines.
    
    Returns:
        tuple: x_range and y_range
    """
    # 1. Create a linspace for x values
    x_values = np.linspace(x_start, x_end, samples)
    y_values = np.array([f(x) for x in x_values])
    
    # 2. Compute the curvature using finite differences
    dx = x_values[1] - x_values[0]
    dydx = np.gradient(y_values, dx)
    d2ydx2 = np.gradient(dydx, dx)
    
    curvature = np.abs(d2ydx2) / (1 + dydx**2)**1.5
    
    # If the function is a simple line, give a tighter domain (around 20) and focus on the y-intercept
    if focus_on_y_intercept and np.all(curvature < 1e-5):
        x_range = (-10, 10)  # Tighter domain around y-intercept
        y_tightened_domain = [f(x) for x in np.linspace(x_range[0], x_range[1], samples)]
        y_range = (min(y_tightened_domain), max(y_tightened_domain))
    else:
        # 3. Identify the x values where curvature is high (e.g. top 10% of curvature values)
        threshold = np.percentile(curvature, 90)
        mask = curvature > threshold
    
        # 4. Set x_range and y_range
        x_range = (x_values[mask].min(), x_values[mask].max())
        y_range = (y_values[mask].min(), y_values[mask].max())
    
    return x_range, y_range



def determine_step_size(value_range, preferred_ticks=10, max_ticks=15):
    """
    Determines a suitable step size based on a given range and a preferred number of ticks.
    
    Args:
        value_range (tuple): Tuple of (min, max) representing the range.
        preferred_ticks (int): Preferred number of ticks for the range.
        max_ticks (int): Maximum allowable number of ticks.
    
    Returns:
        float: Step size for the range.
    """
    
    span = value_range[1] - value_range[0]
    
    # Calculate an initial step
    raw_step = span / preferred_ticks
    
    # Define possible step sizes
    possible_steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000]
    
    # Sort the possible steps based on their difference from raw_step
    sorted_steps = sorted(possible_steps, key=lambda x: abs(x - raw_step))
    
    # Choose the closest step size that doesn't exceed max_ticks
    for step in sorted_steps:
        if span / step <= max_ticks:
            return step
            
    # If no suitable step size found, return the largest one as a fallback
    return possible_steps[-1]

