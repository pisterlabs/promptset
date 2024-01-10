# Import required libraries 
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, State
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from PIL import Image
from openai import OpenAI
import openai
import matplotlib.pyplot as plt
import os 
import numpy as np
import pandas as pd
from tqdm import tqdm
import replicate
import pickle 
import random 



##### To-Do #####
# Actual Live Experiment !!! 
# Adapt plotting functions to new data format (PT, DE, more?)
# Output prompt in live experiments
# Output/visualize original results in live experiments
# Return dataframe in live experiments


# LLAma is suddenly very slow

# Manage installation of not yet installed packages for the user

########################################
# Right now, this helps us to test the live experiment. Later we want the user to enter its API key

# Get openAI API key (previously saved as environmental variable)
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set client
client = OpenAI()
########################################



########################################  Data Import Functions  ########################################



# Load in results of Decoy Effect experiments
DE_probs = pd.read_csv("Output/DE_probs.csv", index_col = 0)

# Load in results and graphs of Prospect Theory experiments
PT_probs = pd.read_csv("Output/PT_probs.csv", index_col = 0)
PT_og_scenario1 = Image.open("Output/PT_og_scenario1.png")
PT_og_scenario2 = Image.open("Output/PT_og_scenario2.png")
PT_og_scenario3 = Image.open("Output/PT_og_scenario3.png")
PT_og_scenario4 = Image.open("Output/PT_og_scenario4.png")

# Second Prospect Theory experiment
PT2_probs = pd.read_csv("Output/PT2_probs.csv", index_col = 0)

# Function for getting data of Sunk Cost Experiment 1
def get_sunk_cost_data_1(selected_temperature, selected_sunk_cost):
    sunk_cost_1 = pd.read_csv('Output/Sunk_cost_experiment_1_with_llama.csv', index_col=0)
    df = sunk_cost_1[(sunk_cost_1['Temperature'] == selected_temperature) & 
                     (sunk_cost_1['Sunk Cost ($)'] == selected_sunk_cost)]
    
    return df

# Function for getting data of Sunk Cost Experiment 2
def get_sunk_cost_data_2(selected_temperature, selected_model):
    df = pd.read_csv('Output/Sunk_cost_experiment_2_with_llama.csv', index_col=0)
    df = df[(df['Temperature'] == selected_temperature) & 
            (df['Model'] == selected_model) |
            (df['Model'] == 'Real Experiment')]
    
    return df

# Function for getting data of Loss Aversion Experiment
def get_loss_aversion_data(selected_temperature):
    df = pd.read_csv('Output/Loss_aversion_experiment_with_llama.csv', index_col=0)
    df = df[(df['Temperature'] == selected_temperature)|
            (df['Model'] == 'Real Experiment')] 
    
    return df   
        
        

########################################  Data Plotting Functions  ########################################

# Function for plotting results of decoy effect/prospect theory experiments
def plot_results(model, priming, df, scenario):
    
    # Get dataframe as specified by user (subset of df)
    df = df[(df['Model'] == model) & (df['Priming'] == priming) & (df['Scenario'] == scenario)]
    # Transpose for plotting
    df = df.transpose()
    
    # Get number of observations per temperature value
    n_observations = df.loc["Obs."]
    
    # Get temperature values
    temperature = df.loc["Temp"]

    fig = go.Figure(data=[
        go.Bar(
            name="p(A)", 
            x=temperature, 
            y=df.loc["p(A)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br>Observations: %{customdata}<extra></extra>",
            marker=dict(color="#e9724d"),
        ),
        go.Bar(
            name="p(B)", 
            x=temperature, 
            y=df.loc["p(B)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br> Observations: %{customdata}<extra></extra>",
            marker=dict(color="#868686"),
            
        ),
        go.Bar(
            name="p(C)", 
            x=temperature, 
            y=df.loc["p(C)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br> Observations: %{customdata}<extra></extra>",
            marker=dict(color="#92cad1"),
        )
    ])

    fig.update_layout(
    barmode = 'group',
    xaxis = dict(
        tickmode = 'array',
        tickvals = temperature,
        ticktext = temperature,
        title = "Temperature",  
        title_font=dict(size=18),  
    ),
    yaxis = dict(
        title="Probability (%)",  
        title_font=dict(size=18), 
    ),
    title = dict(
        text="Distribution of answers per temperature value",
        x = 0.5, # Center alignment horizontally
        y = 0.87,  # Vertical alignment
        font=dict(size=22),  
    ),
    legend = dict(
        title = dict(text="Probabilities"),
    ),
    bargap = 0.3  # Gap between temperature values
)
    return fig

# Function to plot individual experiment results of PT & DE
def plot_results_individual_recreate(df):
    
    # Get number of observations per temperature value
    n_observations = df.loc["Obs."]
    
    # Get temperature values
    temperature = df.loc["Temp"]

    # Get model
    model = df.loc["Model"][0]
    if model == "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3":
        model = "llama-2-70b-chat"

    # Get experiment id
    experiment_id = df.loc["Experiment"][0]

    fig = go.Figure(data=[
        go.Bar(
            name="p(A)", 
            x=temperature, 
            y=df.loc["p(A)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br>Total Observations: %{customdata}<extra></extra>",
            marker=dict(color="#e9724d"),
        ),
        go.Bar(
            name="p(B)", 
            x=temperature, 
            y=df.loc["p(B)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br>Total Observations: %{customdata}<extra></extra>",
            marker=dict(color="#868686"),
            
        ),
        go.Bar(
            name="p(C)", 
            x=temperature, 
            y=df.loc["p(C)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br>Total Observations: %{customdata}<extra></extra>",
            marker=dict(color="#92cad1"),
        )
    ])

    fig.update_layout(
    barmode = 'group',
    xaxis = dict(
        tickmode = 'array',
        tickvals = temperature,
        ticktext = temperature,
        title = "Temperature",  
        title_font=dict(size=18),  
    ),
    yaxis = dict(
        title="Probability (%)",  
        title_font=dict(size=18), 
    ),
    title = dict(
        text= f"Distribution of answers for experiment {experiment_id} using model {model}",
        x = 0.5, # Center alignment horizontally
        y = 0.87,  # Vertical alignment
        font=dict(size=22),  
    ),
    legend = dict(
        title = dict(text="Probabilities"),
    ),
    bargap = 0.3  # Gap between temperature values
)
    return fig

# Function to plot results of individual experiment
def plot_results_individual(df):
    
    # Get number of observations per temperature value
    n_observations = df.loc["Obs."]
    
    # Get temperature values
    temperature = df.loc["Temp"]

    # Get model
    model = df.loc["Model"][0]

    # Rename for better readability
    if model == "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3":
        model = "llama-2-70b-chat"


    fig = go.Figure(data=[
        go.Bar(
            name="p(A)", 
            x=temperature, 
            y=df.loc["p(A)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br>Observations: %{customdata}<extra></extra>",
            marker=dict(color="#e9724d"),
        ),
        go.Bar(
            name="p(B)", 
            x=temperature, 
            y=df.loc["p(B)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br> Observations: %{customdata}<extra></extra>",
            marker=dict(color="#868686"),
            
        ),
        go.Bar(
            name="p(C)", 
            x=temperature, 
            y=df.loc["p(C)"],
            customdata = n_observations,
            hovertemplate="Temperature: %{x}<br>Probability: %{y:.2f}%<br> Observations: %{customdata}<extra></extra>",
            marker=dict(color="#92cad1"),
        )
    ])

    fig.update_layout(
    barmode = 'group',
    xaxis = dict(
        tickmode = 'array',
        tickvals = temperature,
        ticktext = temperature,
        title = "Temperature",  
        title_font=dict(size=18),  
    ),
    yaxis = dict(
        title="Probability (%)",  
        title_font=dict(size=18), 
    ),
    title = dict(
        text= f"Results of experiment with {model}",
        x = 0.5, # Center alignment horizontally
        y = 0.87,  # Vertical alignment
        font=dict(size=22),  
    ),
    legend = dict(
        title = dict(text="Probabilities"),
    ),
    bargap = 0.3  # Gap between temperature values
)
    return fig

# Function for plotting Sunk Cost Experiment 1
def plot_sunk_cost_1(selected_temperature, selected_sunk_cost):
    df_sunk_cost = get_sunk_cost_data_1(selected_temperature, selected_sunk_cost)
    
    # Create a bar plot
    fig_sunk_cost = go.Figure()
    fig_sunk_cost.add_trace(go.Bar(
        x=df_sunk_cost['Model'],
        y=df_sunk_cost['Share Theater Performance'],
        name='Share Theater Performance',
        hovertemplate="Theater Performance: %{y:.2f}<extra></extra>"
    ))
    fig_sunk_cost.add_trace(go.Bar(
        x=df_sunk_cost['Model'],
        y=df_sunk_cost['Share Rock Concert'],
        name='Share Rock Concert',
        hovertemplate="Rock Concert: %{y:.2f}<extra></extra>"
    ))
    
    fig_sunk_cost.update_layout(
        barmode='group',
        xaxis=dict(title='Model'),
        yaxis=dict(title='Share', range=[0, 1.1]),
        title=dict(text=f"Shares for Answer Options (Sunk Cost: ${selected_sunk_cost}, Temperature: {selected_temperature})",
                   x=0.45),
        legend=dict(),
        bargap=0.3  # Gap between models
    )

    return fig_sunk_cost

# Function for plotting Sunk Cost Experiment 2
def plot_sunk_cost_2(selected_temperature, selected_model):
    df = get_sunk_cost_data_2(selected_temperature, selected_model)
    
    cols_to_select = df.columns.tolist().index('$0')
    end_col = df.columns.tolist().index('-$55')

    # Get unique models and prompts
    models = df['Model'].unique()
    prompts = df['Prompt'].unique()

    # Set the width of the bars
    bar_width = 0.1

    fig = go.Figure()

    # Iterate over each model
    for model in models:
        if model != 'Real Experiment':
            for i, prompt in enumerate(prompts):
                subset = df[df['Prompt'] == prompt]

                if not subset.empty:
                    fig.add_trace(go.Bar(
                        x=np.arange(len(df.columns[cols_to_select:end_col+1])) + (i * bar_width),
                        y=subset.iloc[0, cols_to_select:end_col+1].values,
                        width=bar_width,
                        name=f'Answer Option Order {i + 1}',
                        marker=dict(color=f'rgba({i * 50}, 0, 255, 0.6)'),
                        hovertemplate="%{y:.2f}",
                    ))
        elif model == 'Real Experiment':
                fig.add_trace(go.Bar(
                            x=np.arange(len(df.columns[cols_to_select:end_col+1])) + ((len(prompts)-1) * bar_width),
                            y=df.iloc[-1, cols_to_select:end_col+1].values,
                            width=bar_width,
                            name='Real Results',
                            marker=dict(color='rgba(0, 0, 0, 0.3)'),
                            hovertemplate="%{y:.2f}",
                        ))


    fig.update_layout(
        barmode='group',
        xaxis=dict(tickvals=np.arange(len(df.columns[cols_to_select:end_col+1])) + ((len(prompts) - 1) / 2 * bar_width),
                ticktext=['$0', '$20', '$20 plus interest', '$75', '-$55']),
        yaxis=dict(title='Share', range=[0, 1.1]),
        title=dict(text=f'Shares for Answer Options (Model: {selected_model}, Temperature: {selected_temperature})',
                   x=0.45),
        legend=dict(),
        bargap=0.3  # Gap between bars
    )

    return fig

# Function for plotting Loss Aversion Experiment
def plot_loss_aversion(selected_temperature):
    df = get_loss_aversion_data(selected_temperature)
    
    # Extract unique models
    models = df['Model'].unique()

    # Set up figure
    fig = go.Figure()

    # Plotting bars for 'B' for each prompt
    for i, prompt in enumerate(df['Scenario'].unique()):
        values_B = df[df['Scenario'] == prompt]['B']
        scenario_label = 'Scenario with gains' if prompt == 1 else 'Scenario with losses'
        fig.add_trace(go.Bar(
            x=models,
            y=values_B,
            name=scenario_label,
            offsetgroup=i,
            hovertemplate="%{y:.2f}",
        ))

    # Update layout
    fig.update_layout(
        barmode='group',
        xaxis=dict(tickmode='array', tickvals=list(range(len(models))), ticktext=models),
        yaxis=dict(title='Shares for "B"'),
        title=dict(text='Shares for "B" (risk-seeking option) by Model and Scenario',
                   x=0.45),
        bargap=0.6  # Gap between bars
    )
    
    return fig

########################################  Prompts  ########################################

### Prospect Theory ###
PT_prompt_1 = """Mr. A was given tickets involving the World Series. He won 50$ in one lottery and $25 in the other. 
          Mr. B was given a ticket to a single, larger World Series lottery. He won $75. Based solely on this information, Who is happier? 
          A: Mister A
          B: Mister B
          C: No difference.         
          Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_2 = """Mr. A received a letter from the IRS saying that he made a minor arithmetical mistake on his tax return and owed $100. 
         He received a similar letter the same day from his state income tax authority saying he owed $50. There were no other repercussions from either mistake. 
         Mr. B received a letter from the IRS saying that he made a minor arithmetical mistake on his tax return and owed $150. There were no other repercussions from his mistake. 
         Based solely on this information, who was more upset? 
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_3 = """Mr. A bought his first New York State lottery ticket and won $100. Also, in a freak accident, he damaged the rug in his apartment and had to pay the landlord $80.
         Mr. B bought his first New York State lottery ticket and won $20. Based solely on this information, who is happier? 
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_4 = """Mr. A's car was damaged in a parking lot. He had to spend $200 to repair the damage. The same day the car was damaged, he won $25 in the office football pool.
         Mr. B's car was damaged in a parking lot. He had to spend $175 to repair the damage. Based solely on this information, who is more upset?
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_5 = """You are a market researcher and focus on Prospect Theory and Mental Accounting. In a survey you are presented the following situation: 
          Mr. A was given tickets involving the World Series. He won 50$ in one lottery and 25$ in the other. 
          Mr. B was given a ticket to a single, larger World Series lottery. He won 75$. Based solely on this information, who is happier?
          A: Mister A
          B: Mister B
          C: No difference.
          Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_6 = """You are a market researcher and focus on Prospect Theory and Mental Accounting. In a survey you are presented the following situation:
         Mr. A received a letter from the IRS saying that he made a minor arithmetical mistake on his tax return and owed $100. 
         He received a similar letter the same day from his state income tax authority saying he owed $50. There were no other repercussions from either mistake. 
         Mr. B received a letter from the IRS saying that he made a minor arithmetical mistake on his tax return and owed $150. There were no other repercussions from his mistake. 
         Based solely on this information, who was more upset? 
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_7 = """You are a market researcher and focus on Prospect Theory and Mental Accounting. In a survey you are presented the following situation:
         Mr. A bought his first New York State lottery ticket and won $100. Also, in a freak accident, he damaged the rug in his apartment and had to pay the landlord $80.
         Mr. B bought his first New York State lottery ticket and won $20? Based solely on this information, who is happier?
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
PT_prompt_8 = """You are a market researcher and focus on Prospect Theory and Mental Accounting. In a survey you are presented the following situation:
         Mr. A's car was damaged in a parking lot. He had to spend $200 to repair the damage. The same day the car was damaged, he won $25 in the office football pool.
         Mr. B's car was damaged in a parking lot. He had to spend $175 to repair the damage. Based solely on this information, who is more upset?
         A: Mister A
         B: Mister B
         C: No difference.
         Which option would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""

### Decoy Effect ###
DE_prompt_1 = """You are presented with the following subscription alternatives for the "The Economist" magazine:
        A: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$.
        B: One-year subscription to the print edition of The Economist, priced at 125$.
        C: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
        Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_2 = """You are presented with the following subscription alternatives for the "The Economist" magazine:
        A: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$.
        B: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$. 
        Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_3 = """You are a market researcher that knows about the Decoy Effect in pricing. 
        You are presented with the following subscription alternatives for the "The Economist" magazine:
        A: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$.
        B: One-year subscription to the print edition of The Economist, priced at 125$.
        C: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
        Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_4 = """You are a market researcher that knows about the Decoy Effect in pricing. 
         You are presented with the following subscription alternatives for the "The Economist" magazine:
         A: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$.
         B: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
         Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_5 = """You are presented with the following subscription alternatives for the "The Economist" magazine:
         Q: One-year subscription to the print edition of The Economist, priced at 125$.
         X: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
         Y: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$. 
         Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_6 = """You are presented with the following subscription alternatives for the "The Economist" magazine:
         X: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
         Y: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$. 
         Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_7 = """You are a market researcher that knows about the Decoy Effect in pricing. 
         You are presented with the following subscription alternatives for the "The Economist" magazine:
         Q: One-year subscription to the print edition of The Economist, priced at 125$.
         X: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
         Y: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$. 
         Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""
DE_prompt_8 = """You are a market researcher that knows about the Decoy Effect in pricing. 
         You are presented with the following subscription alternatives for the "The Economist" magazine:
         X: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.
         Y: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$. 
         Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."""

### Prospect Theory 2.0 ###
# Scenario 1
with open("PT2_prompts_1.pkl", "rb") as file:
    PT2_prompts_1 = pickle.load(file)
# Scenario 2
with open("PT2_prompts_2.pkl", "rb") as file:
    PT2_prompts_2 = pickle.load(file)

# Scenario 3
with open("PT2_prompts_3.pkl", "rb") as file:
    PT2_prompts_3 = pickle.load(file)

# Scenario 4
with open("PT2_prompts_4.pkl", "rb") as file:
    PT2_prompts_4 = pickle.load(file)

######################################## Dictionaries  ########################################
    
### Prospect Theory ###
# Original results for Prospect Theory Experiments
PT_p_scenario1 = [f"p(A): {round((56/(56+16+15)*100), 2)}%", f"p(B): {round((16/(56+16+15)*100), 2)}%", f"p(C): {round((15/(56+16+15)*100), 2)}%"]
PT_p_scenario2 = [f"p(A): {round((66/(66+14+7)*100), 2)}%", f"p(B): {round((14/(66+14+7)*100), 2)}%", f"p(C): {round((7/(66+14+7)*100), 2)}%"]
PT_p_scenario3 = [f"p(A): {round((22/(22+61+4)*100), 2)}%", f"p(B): {round((61/(22+61+4)*100), 2)}%", f"p(C): {round((4/(22+61+4)*100), 2)}%"]
PT_p_scenario4 = [f"p(A): {round((19/(19+63+5)*100), 2)}%", f"p(B): {round((63/(19+63+5)*100), 2)}%", f"p(C): {round((5/(19+63+5)*100), 2)}%"]

# Dictionary that returns the literal prompt for a given experiment id (used in function call). key: experiment_id, value: prompt
PT_experiment_prompts_dict = {
    "PT_1_1": PT_prompt_1,
    "PT_1_2": PT_prompt_2,
    "PT_1_3": PT_prompt_3,
    "PT_1_4": PT_prompt_4,
    "PT_1_5": PT_prompt_5,
    "PT_1_6": PT_prompt_6,
    "PT_1_7": PT_prompt_7,
    "PT_1_8": PT_prompt_8,
    "PT_2_1": PT_prompt_1,
    "PT_2_2": PT_prompt_2,
    "PT_2_3": PT_prompt_3,
    "PT_2_4": PT_prompt_4,
    "PT_2_5": PT_prompt_5,
    "PT_2_6": PT_prompt_6,
    "PT_2_7": PT_prompt_7,
    "PT_2_8": PT_prompt_8,
    "PT_3_1": PT_prompt_1,
    "PT_3_2": PT_prompt_2,
    "PT_3_3": PT_prompt_3,
    "PT_3_4": PT_prompt_4,
    "PT_3_5": PT_prompt_5,
    "PT_3_6": PT_prompt_6,
    "PT_3_7": PT_prompt_7,
    "PT_3_8": PT_prompt_8,
}

# It returns the variable name of the prompt that was used in the experiment. key: experiment_id, value: prompt_name
PT_prompt_ids_dict = {
    "PT_1_1": "PT_prompt_1",
    "PT_1_2": "PT_prompt_2",
    "PT_1_3": "PT_prompt_3",
    "PT_1_4": "PT_prompt_4",
    "PT_1_5": "PT_prompt_5",
    "PT_1_6": "PT_prompt_6",
    "PT_1_7": "PT_prompt_7",
    "PT_1_8": "PT_prompt_8",
    "PT_2_1": "PT_prompt_1",
    "PT_2_2": "PT_prompt_2",
    "PT_2_3": "PT_prompt_3",
    "PT_2_4": "PT_prompt_4",
    "PT_2_5": "PT_prompt_5",
    "PT_2_6": "PT_prompt_6",
    "PT_2_7": "PT_prompt_7",
    "PT_2_8": "PT_prompt_8",
    "PT_3_1": "PT_prompt_1",
    "PT_3_2": "PT_prompt_2",
    "PT_3_3": "PT_prompt_3",
    "PT_3_4": "PT_prompt_4",
    "PT_3_5": "PT_prompt_5",
    "PT_3_6": "PT_prompt_6",
    "PT_3_7": "PT_prompt_7",
    "PT_3_8": "PT_prompt_8",
}

# Dictionary to look up which model to use for a given experiment id (used in function call). key: experiment id, value: model name
PT_model_dict = {
    "PT_1_1": "gpt-3.5-turbo",
    "PT_1_2": "gpt-3.5-turbo",
    "PT_1_3": "gpt-3.5-turbo",
    "PT_1_4": "gpt-3.5-turbo",
    "PT_1_5": "gpt-3.5-turbo",
    "PT_1_6": "gpt-3.5-turbo",
    "PT_1_7": "gpt-3.5-turbo",
    "PT_1_8": "gpt-3.5-turbo",
    "PT_2_1": "gpt-4-1106-preview",
    "PT_2_2": "gpt-4-1106-preview",
    "PT_2_3": "gpt-4-1106-preview",
    "PT_2_4": "gpt-4-1106-preview",
    "PT_2_5": "gpt-4-1106-preview",
    "PT_2_6": "gpt-4-1106-preview",
    "PT_2_7": "gpt-4-1106-preview",
    "PT_2_8": "gpt-4-1106-preview",
    "PT_3_1": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_2": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_3": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_4": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_5": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_6": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_7": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "PT_3_8": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    }

# Dictionary to look up the original results of the experiments. key: experiment id, value: original result
PT_results_dict = {
    "PT_1_1": PT_p_scenario1,
    "PT_1_2": PT_p_scenario2,
    "PT_1_3": PT_p_scenario3,
    "PT_1_4": PT_p_scenario4,
    "PT_1_5": PT_p_scenario1,
    "PT_1_6": PT_p_scenario2,
    "PT_1_7": PT_p_scenario3,
    "PT_1_8": PT_p_scenario4,
    "PT_2_1": PT_p_scenario1,
    "PT_2_2": PT_p_scenario2,
    "PT_2_3": PT_p_scenario3,
    "PT_2_4": PT_p_scenario4,
    "PT_2_5": PT_p_scenario1,
    "PT_2_6": PT_p_scenario2,
    "PT_2_7": PT_p_scenario3,
    "PT_2_8": PT_p_scenario4,
    "PT_3_1": PT_p_scenario1,
    "PT_3_2": PT_p_scenario2,
    "PT_3_3": PT_p_scenario3,
    "PT_3_4": PT_p_scenario4,
    "PT_3_5": PT_p_scenario1,
    "PT_3_6": PT_p_scenario2,
    "PT_3_7": PT_p_scenario3,
    "PT_3_8": PT_p_scenario4,
}

# Dictionary to look up the scenario number of a given experiment ID. key: experiment id, value: scenario number
PT_scenario_dict = {
    "PT_1_1": 1,
    "PT_1_2": 2,
    "PT_1_3": 3,
    "PT_1_4": 4,
    "PT_1_5": 1,
    "PT_1_6": 2,
    "PT_1_7": 3,
    "PT_1_8": 4,
    "PT_2_1": 1,
    "PT_2_2": 2,
    "PT_2_3": 3,
    "PT_2_4": 4,
    "PT_2_5": 1,
    "PT_2_6": 2,
    "PT_2_7": 3,
    "PT_2_8": 4,
    "PT_3_1": 1,
    "PT_3_2": 2,
    "PT_3_3": 3,
    "PT_3_4": 4,
    "PT_3_5": 1,
    "PT_3_6": 2,
    "PT_3_7": 3,
    "PT_3_8": 4,
}   

# Dictionary to look up, whether an experiment used a primed or unprimed prompt. key: experiment id, value: 1 if primed, 0 if unprimed
PT_priming_dict = {
    "PT_1_1": 0,
    "PT_1_2": 0,
    "PT_1_3": 0,
    "PT_1_4": 0,
    "PT_1_5": 1,
    "PT_1_6": 1,
    "PT_1_7": 1,
    "PT_1_8": 1,
    "PT_2_1": 0,
    "PT_2_2": 0,
    "PT_2_3": 0,
    "PT_2_4": 0,
    "PT_2_5": 1,
    "PT_2_6": 1,
    "PT_2_7": 1,
    "PT_2_8": 1,
    "PT_3_1": 0,
    "PT_3_2": 0,
    "PT_3_3": 0,
    "PT_3_4": 0,
    "PT_3_5": 1,
    "PT_3_6": 1,
    "PT_3_7": 1,
    "PT_3_8": 1,
}

# Dictionary to look up original results of the Prospect Theory experiments. Key: experiment id, value: original results
PT_results_dict = {
    "PT_1_1": PT_p_scenario1,
    "PT_1_2": PT_p_scenario2,
    "PT_1_3": PT_p_scenario3,
    "PT_1_4": PT_p_scenario4,
    "PT_1_5": PT_p_scenario1,
    "PT_1_6": PT_p_scenario2,
    "PT_1_7": PT_p_scenario3,
    "PT_1_8": PT_p_scenario4,
    "PT_2_1": PT_p_scenario1,
    "PT_2_2": PT_p_scenario2,
    "PT_2_3": PT_p_scenario3,
    "PT_2_4": PT_p_scenario4,
    "PT_2_5": PT_p_scenario1,
    "PT_2_6": PT_p_scenario2,
    "PT_2_7": PT_p_scenario3,
    "PT_2_8": PT_p_scenario4,
    "PT_3_1": PT_p_scenario1,
    "PT_3_2": PT_p_scenario2,
    "PT_3_3": PT_p_scenario3,
    "PT_3_4": PT_p_scenario4,
    "PT_3_5": PT_p_scenario1,
    "PT_3_6": PT_p_scenario2,
    "PT_3_7": PT_p_scenario3,
    "PT_3_8": PT_p_scenario4,
    }


### Prospect Theory 2.0 ###
# Dictionary to look up prompt for a given experiment id. key: experiment id, value: prompt
PT2_experiment_prompts_dict = {
    "PT2_1_1_1": PT2_prompts_1[0],
    "PT2_1_1_2": PT2_prompts_1[1],
    "PT2_1_1_3": PT2_prompts_1[2],
    "PT2_1_1_4": PT2_prompts_1[3],
    "PT2_1_1_5": PT2_prompts_1[4],
    "PT2_1_1_6": PT2_prompts_1[5],
    "PT2_2_1_1": PT2_prompts_2[0],
    "PT2_2_1_2": PT2_prompts_2[1],
    "PT2_2_1_3": PT2_prompts_2[2],
    "PT2_2_1_4": PT2_prompts_2[3],
    "PT2_2_1_5": PT2_prompts_2[4],
    "PT2_2_1_6": PT2_prompts_2[5],
    "PT2_3_1_1": PT2_prompts_3[0],
    "PT2_3_1_2": PT2_prompts_3[1],
    "PT2_3_1_3": PT2_prompts_3[2],
    "PT2_3_1_4": PT2_prompts_3[3],
    "PT2_3_1_5": PT2_prompts_3[4],
    "PT2_3_1_6": PT2_prompts_3[5],
    "PT2_4_1_1": PT2_prompts_4[0],
    "PT2_4_1_2": PT2_prompts_4[1],
    "PT2_4_1_3": PT2_prompts_4[2],
    "PT2_4_1_4": PT2_prompts_4[3],
    "PT2_4_1_5": PT2_prompts_4[4],
    "PT2_4_1_6": PT2_prompts_4[5],
    "PT2_1_2_1": PT2_prompts_1[0],
    "PT2_1_2_2": PT2_prompts_1[1],
    "PT2_1_2_3": PT2_prompts_1[2],
    "PT2_1_2_4": PT2_prompts_1[3],
    "PT2_1_2_5": PT2_prompts_1[4],
    "PT2_1_2_6": PT2_prompts_1[5],
    "PT2_2_2_1": PT2_prompts_2[0],
    "PT2_2_2_2": PT2_prompts_2[1],
    "PT2_2_2_3": PT2_prompts_2[2],
    "PT2_2_2_4": PT2_prompts_2[3],
    "PT2_2_2_5": PT2_prompts_2[4],
    "PT2_2_2_6": PT2_prompts_2[5],
    "PT2_3_2_1": PT2_prompts_3[0],
    "PT2_3_2_2": PT2_prompts_3[1],
    "PT2_3_2_3": PT2_prompts_3[2],
    "PT2_3_2_4": PT2_prompts_3[3],
    "PT2_3_2_5": PT2_prompts_3[4],
    "PT2_3_2_6": PT2_prompts_3[5],
    "PT2_4_2_1": PT2_prompts_4[0],
    "PT2_4_2_2": PT2_prompts_4[1],
    "PT2_4_2_3": PT2_prompts_4[2],
    "PT2_4_2_4": PT2_prompts_4[3],
    "PT2_4_2_5": PT2_prompts_4[4],
    "PT2_4_2_6": PT2_prompts_4[5],
    "PT2_1_3_1": PT2_prompts_1[0],
    "PT2_1_3_2": PT2_prompts_1[1],
    "PT2_1_3_3": PT2_prompts_1[2],
    "PT2_1_3_4": PT2_prompts_1[3],
    "PT2_1_3_5": PT2_prompts_1[4],
    "PT2_1_3_6": PT2_prompts_1[5],
    "PT2_2_3_1": PT2_prompts_2[0],
    "PT2_2_3_2": PT2_prompts_2[1],
    "PT2_2_3_3": PT2_prompts_2[2],
    "PT2_2_3_4": PT2_prompts_2[3],
    "PT2_2_3_5": PT2_prompts_2[4],
    "PT2_2_3_6": PT2_prompts_2[5],
    "PT2_3_3_1": PT2_prompts_3[0],
    "PT2_3_3_2": PT2_prompts_3[1],
    "PT2_3_3_3": PT2_prompts_3[2],
    "PT2_3_3_4": PT2_prompts_3[3],
    "PT2_3_3_5": PT2_prompts_3[4],
    "PT2_3_3_6": PT2_prompts_3[5],
    "PT2_4_3_1": PT2_prompts_4[0],
    "PT2_4_3_2": PT2_prompts_4[1],
    "PT2_4_3_3": PT2_prompts_4[2],
    "PT2_4_3_4": PT2_prompts_4[3],
    "PT2_4_3_5": PT2_prompts_4[4],
    "PT2_4_3_6": PT2_prompts_4[5],
}

# Dictionary to look up the original results for a given experiment id. key: experiment id, value: original answer probabilities
PT2_results_dict = {
    "PT2_1_1_1": PT_p_scenario1,
    "PT2_1_1_2": PT_p_scenario1,
    "PT2_1_1_3": PT_p_scenario1,
    "PT2_1_1_4": PT_p_scenario1,
    "PT2_1_1_5": PT_p_scenario1,
    "PT2_1_1_6": PT_p_scenario1,
    "PT2_2_1_1": PT_p_scenario2,
    "PT2_2_1_2": PT_p_scenario2,
    "PT2_2_1_3": PT_p_scenario2,
    "PT2_2_1_4": PT_p_scenario2,
    "PT2_2_1_5": PT_p_scenario2,
    "PT2_2_1_6": PT_p_scenario2,
    "PT2_3_1_1": PT_p_scenario3,
    "PT2_3_1_2": PT_p_scenario3,
    "PT2_3_1_3": PT_p_scenario3,
    "PT2_3_1_4": PT_p_scenario3,
    "PT2_3_1_5": PT_p_scenario3,
    "PT2_3_1_6": PT_p_scenario3,
    "PT2_4_1_1": PT_p_scenario4,
    "PT2_4_1_2": PT_p_scenario4,
    "PT2_4_1_3": PT_p_scenario4,
    "PT2_4_1_4": PT_p_scenario4,
    "PT2_4_1_5": PT_p_scenario4,
    "PT2_4_1_6": PT_p_scenario4,
    "PT2_1_2_1": PT_p_scenario1,
    "PT2_1_2_2": PT_p_scenario1,
    "PT2_1_2_3": PT_p_scenario1,
    "PT2_1_2_4": PT_p_scenario1,
    "PT2_1_2_5": PT_p_scenario1,
    "PT2_1_2_6": PT_p_scenario1,
    "PT2_2_2_1": PT_p_scenario2,
    "PT2_2_2_2": PT_p_scenario2,
    "PT2_2_2_3": PT_p_scenario2,
    "PT2_2_2_4": PT_p_scenario2,
    "PT2_2_2_5": PT_p_scenario2,
    "PT2_2_2_6": PT_p_scenario2,
    "PT2_3_2_1": PT_p_scenario3,
    "PT2_3_2_2": PT_p_scenario3,
    "PT2_3_2_3": PT_p_scenario3,
    "PT2_3_2_4": PT_p_scenario3,
    "PT2_3_2_5": PT_p_scenario3,
    "PT2_3_2_6": PT_p_scenario3,
    "PT2_4_2_1": PT_p_scenario4,
    "PT2_4_2_2": PT_p_scenario4,
    "PT2_4_2_3": PT_p_scenario4,
    "PT2_4_2_4": PT_p_scenario4,
    "PT2_4_2_5": PT_p_scenario4,
    "PT2_4_2_6": PT_p_scenario4,
    "PT2_1_3_1": PT_p_scenario1,
    "PT2_1_3_2": PT_p_scenario1,
    "PT2_1_3_3": PT_p_scenario1,
    "PT2_1_3_4": PT_p_scenario1,   
    "PT2_1_3_5": PT_p_scenario1,
    "PT2_1_3_6": PT_p_scenario1,
    "PT2_2_3_1": PT_p_scenario2,
    "PT2_2_3_2": PT_p_scenario2,
    "PT2_2_3_3": PT_p_scenario2,
    "PT2_2_3_4": PT_p_scenario2,
    "PT2_2_3_5": PT_p_scenario2,
    "PT2_2_3_6": PT_p_scenario2,
    "PT2_3_3_1": PT_p_scenario3,
    "PT2_3_3_2": PT_p_scenario3,
    "PT2_3_3_3": PT_p_scenario3,
    "PT2_3_3_4": PT_p_scenario3,
    "PT2_3_3_5": PT_p_scenario3,
    "PT2_3_3_6": PT_p_scenario3,
    "PT2_4_3_1": PT_p_scenario4,
    "PT2_4_3_2": PT_p_scenario4,
    "PT2_4_3_3": PT_p_scenario4,
    "PT2_4_3_4": PT_p_scenario4,
    "PT2_4_3_5": PT_p_scenario4,
    "PT2_4_3_6": PT_p_scenario4,
}

# Dictionary to look up which model to use for a given experiment id. key: experiment id, value: model name
PT2_model_dict = {
    "PT2_1_1_1": "gpt-3.5-turbo",  
    "PT2_1_1_2": "gpt-3.5-turbo",
    "PT2_1_1_3": "gpt-3.5-turbo",
    "PT2_1_1_4": "gpt-3.5-turbo",
    "PT2_1_1_5": "gpt-3.5-turbo",
    "PT2_1_1_6": "gpt-3.5-turbo",
    "PT2_2_1_1": "gpt-3.5-turbo",
    "PT2_2_1_2": "gpt-3.5-turbo",
    "PT2_2_1_3": "gpt-3.5-turbo",
    "PT2_2_1_4": "gpt-3.5-turbo",
    "PT2_2_1_5": "gpt-3.5-turbo",
    "PT2_2_1_6": "gpt-3.5-turbo",
    "PT2_3_1_1": "gpt-3.5-turbo",
    "PT2_3_1_2": "gpt-3.5-turbo",
    "PT2_3_1_3": "gpt-3.5-turbo",
    "PT2_3_1_4": "gpt-3.5-turbo",
    "PT2_3_1_5": "gpt-3.5-turbo",
    "PT2_3_1_6": "gpt-3.5-turbo",
    "PT2_4_1_1": "gpt-3.5-turbo",
    "PT2_4_1_2": "gpt-3.5-turbo",
    "PT2_4_1_3": "gpt-3.5-turbo",
    "PT2_4_1_4": "gpt-3.5-turbo",
    "PT2_4_1_5": "gpt-3.5-turbo",
    "PT2_4_1_6": "gpt-3.5-turbo",
    "PT2_1_2_1": "gpt-4-1106-preview",
    "PT2_1_2_2": "gpt-4-1106-preview",
    "PT2_1_2_3": "gpt-4-1106-preview",
    "PT2_1_2_4": "gpt-4-1106-preview",
    "PT2_1_2_5": "gpt-4-1106-preview",
    "PT2_1_2_6": "gpt-4-1106-preview",
    "PT2_2_2_1": "gpt-4-1106-preview",
    "PT2_2_2_2": "gpt-4-1106-preview",
    "PT2_2_2_3": "gpt-4-1106-preview",
    "PT2_2_2_4": "gpt-4-1106-preview",
    "PT2_2_2_5": "gpt-4-1106-preview",
    "PT2_2_2_6": "gpt-4-1106-preview",
    "PT2_3_2_1": "gpt-4-1106-preview",
    "PT2_3_2_2": "gpt-4-1106-preview",
    "PT2_3_2_3": "gpt-4-1106-preview",
    "PT2_3_2_4": "gpt-4-1106-preview",
    "PT2_3_2_5": "gpt-4-1106-preview",
    "PT2_3_2_6": "gpt-4-1106-preview",
    "PT2_4_2_1": "gpt-4-1106-preview",
    "PT2_4_2_2": "gpt-4-1106-preview",
    "PT2_4_2_3": "gpt-4-1106-preview",
    "PT2_4_2_4": "gpt-4-1106-preview",
    "PT2_4_2_5": "gpt-4-1106-preview",
    "PT2_4_2_6": "gpt-4-1106-preview",
    "PT2_1_3_1": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_1_3_2": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_1_3_3": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_1_3_4": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_1_3_5": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_1_3_6": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_1": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_2": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_3": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_4": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_5": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_2_3_6": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_1": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_2": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_3": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_4": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_5": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_3_3_6": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_1": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_2": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_3": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_4": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_5": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    "PT2_4_3_6": "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
    }

# Dictionary to look up scenario number for a given experiment id. key: experiment id, value: scenario number
PT2_scenario_dict = {
    "PT2_1_1_1": 1,
    "PT2_1_1_2": 1,
    "PT2_1_1_3": 1,
    "PT2_1_1_4": 1,
    "PT2_1_1_5": 1,
    "PT2_1_1_6": 1,
    "PT2_2_1_1": 2,
    "PT2_2_1_2": 2,
    "PT2_2_1_3": 2,
    "PT2_2_1_4": 2,
    "PT2_2_1_5": 2,
    "PT2_2_1_6": 2,
    "PT2_3_1_1": 3,
    "PT2_3_1_2": 3,
    "PT2_3_1_3": 3,
    "PT2_3_1_4": 3,
    "PT2_3_1_5": 3,
    "PT2_3_1_6": 3,
    "PT2_4_1_1": 4,
    "PT2_4_1_2": 4,
    "PT2_4_1_3": 4,
    "PT2_4_1_4": 4,
    "PT2_4_1_5": 4,
    "PT2_4_1_6": 4,
    "PT2_1_2_1": 1,
    "PT2_1_2_2": 1,
    "PT2_1_2_3": 1,
    "PT2_1_2_4": 1,
    "PT2_1_2_5": 1,
    "PT2_1_2_6": 1,
    "PT2_2_2_1": 2,
    "PT2_2_2_2": 2, 
    "PT2_2_2_3": 2,
    "PT2_2_2_4": 2,
    "PT2_2_2_5": 2,
    "PT2_2_2_6": 2,
    "PT2_3_2_1": 3,
    "PT2_3_2_2": 3,
    "PT2_3_2_3": 3,
    "PT2_3_2_4": 3,
    "PT2_3_2_5": 3,
    "PT2_3_2_6": 3,
    "PT2_4_2_1": 4,
    "PT2_4_2_2": 4,
    "PT2_4_2_3": 4,
    "PT2_4_2_4": 4,
    "PT2_4_2_5": 4,
    "PT2_4_2_6": 4,
    "PT2_1_3_1": 1,
    "PT2_1_3_2": 1,
    "PT2_1_3_3": 1,
    "PT2_1_3_4": 1,
    "PT2_1_3_5": 1,
    "PT2_1_3_6": 1,
    "PT2_2_3_1": 2,
    "PT2_2_3_2": 2,
    "PT2_2_3_3": 2,
    "PT2_2_3_4": 2,
    "PT2_2_3_5": 2,
    "PT2_2_3_6": 2,
    "PT2_3_3_1": 3,
    "PT2_3_3_2": 3,
    "PT2_3_3_3": 3,
    "PT2_3_3_4": 3,
    "PT2_3_3_5": 3,
    "PT2_3_3_6": 3,
    "PT2_4_3_1": 4,
    "PT2_4_3_2": 4,
    "PT2_4_3_3": 4,
    "PT2_4_3_4": 4,
    "PT2_4_3_5": 4,
    "PT2_4_3_6": 4,
}

# Dictionary to look up scenario configuration based on experiment id. key: experiment id, value: scenario configuration
PT2_configuration_dict = {
    "PT2_1_1_1": 1,
    "PT2_1_1_2": 2,
    "PT2_1_1_3": 3,
    "PT2_1_1_4": 4,
    "PT2_1_1_5": 5,
    "PT2_1_1_6": 6,
    "PT2_2_1_1": 1,
    "PT2_2_1_2": 2,
    "PT2_2_1_3": 3,
    "PT2_2_1_4": 4,
    "PT2_2_1_5": 5,
    "PT2_2_1_6": 6,
    "PT2_3_1_1": 1,
    "PT2_3_1_2": 2,
    "PT2_3_1_3": 3,
    "PT2_3_1_4": 4,
    "PT2_3_1_5": 5,
    "PT2_3_1_6": 6,
    "PT2_4_1_1": 1,
    "PT2_4_1_2": 2,
    "PT2_4_1_3": 3,
    "PT2_4_1_4": 4,
    "PT2_4_1_5": 5,
    "PT2_4_1_6": 6,
    "PT2_1_2_1": 1,
    "PT2_1_2_2": 2,
    "PT2_1_2_3": 3,
    "PT2_1_2_4": 4,
    "PT2_1_2_5": 5,
    "PT2_1_2_6": 6,
    "PT2_2_2_1": 1,
    "PT2_2_2_2": 2,
    "PT2_2_2_3": 3,
    "PT2_2_2_4": 4,
    "PT2_2_2_5": 5,
    "PT2_2_2_6": 6,
    "PT2_3_2_1": 1,
    "PT2_3_2_2": 2,
    "PT2_3_2_3": 3,
    "PT2_3_2_4": 4,
    "PT2_3_2_5": 5,
    "PT2_3_2_6": 6,
    "PT2_4_2_1": 1,
    "PT2_4_2_2": 2,
    "PT2_4_2_3": 3,
    "PT2_4_2_4": 4,
    "PT2_4_2_5": 5,
    "PT2_4_2_6": 6,
    "PT2_1_3_1": 1,
    "PT2_1_3_2": 2,
    "PT2_1_3_3": 3,
    "PT2_1_3_4": 4,
    "PT2_1_3_5": 5,
    "PT2_1_3_6": 6,
    "PT2_2_3_1": 1,
    "PT2_2_3_2": 2,
    "PT2_2_3_3": 3,
    "PT2_2_3_4": 4,
    "PT2_2_3_5": 5,
    "PT2_2_3_6": 6,
    "PT2_3_3_1": 1,
    "PT2_3_3_2": 2,
    "PT2_3_3_3": 3,
    "PT2_3_3_4": 4,
    "PT2_3_3_5": 5,
    "PT2_3_3_6": 6,
    "PT2_4_3_1": 1,
    "PT2_4_3_2": 2,
    "PT2_4_3_3": 3,
    "PT2_4_3_4": 4,
    "PT2_4_3_5": 5,
    "PT2_4_3_6": 6,
}


### Decoy Effect ###
# Dictionary that returns the literal prompt for a given experiment id (used in function call). key: experiment_id, value: prompt
DE_experiment_prompts_dict = {
    "DE_1_1": DE_prompt_1,
    "DE_1_2": DE_prompt_2,
    "DE_1_3": DE_prompt_3,
    "DE_1_4": DE_prompt_4,
    "DE_1_5": DE_prompt_5,
    "DE_1_6": DE_prompt_6,
    "DE_1_7": DE_prompt_7,
    "DE_1_8": DE_prompt_8,
    "DE_2_1": DE_prompt_1,
    "DE_2_2": DE_prompt_2,
    "DE_2_3": DE_prompt_3,
    "DE_2_4": DE_prompt_4,
    "DE_2_5": DE_prompt_5,
    "DE_2_6": DE_prompt_6,
    "DE_2_7": DE_prompt_7,
    "DE_2_8": DE_prompt_8,
    "DE_3_1": DE_prompt_1,
    "DE_3_2": DE_prompt_2,
    "DE_3_3": DE_prompt_3,
    "DE_3_4": DE_prompt_4,
    "DE_3_5": DE_prompt_5,
    "DE_3_6": DE_prompt_6,
    "DE_3_7": DE_prompt_7,
    "DE_3_8": DE_prompt_8,
}

# It returns the variable name of the prompt that was used in the experiment. key: experiment_id, value: prompt_name
DE_prompt_ids_dict = {
    "DE_1_1": "DE_prompt_1",
    "DE_1_2": "DE_prompt_2",
    "DE_1_3": "DE_prompt_3",
    "DE_1_4": "DE_prompt_4",
    "DE_1_5": "DE_prompt_5",
    "DE_1_6": "DE_prompt_6",
    "DE_1_7": "DE_prompt_7",
    "DE_1_8": "DE_prompt_8",
    "DE_2_1": "DE_prompt_1",
    "DE_2_2": "DE_prompt_2",
    "DE_2_3": "DE_prompt_3",
    "DE_2_4": "DE_prompt_4",
    "DE_2_5": "DE_prompt_5",
    "DE_2_6": "DE_prompt_6",
    "DE_2_7": "DE_prompt_7",
    "DE_2_8": "DE_prompt_8",
    "DE_3_1": "DE_prompt_1",
    "DE_3_2": "DE_prompt_2",
    "DE_3_3": "DE_prompt_3",
    "DE_3_4": "DE_prompt_4",
    "DE_3_5": "DE_prompt_5",
    "DE_3_6": "DE_prompt_6",
    "DE_3_7": "DE_prompt_7",
    "DE_3_8": "DE_prompt_8",
}

# Dictionary to look up which model to use for a given experiment id (used in function call). key: experiment id, value: model name
DE_model_dict = {
    "DE_1_1": "gpt-3.5-turbo",
    "DE_1_2": "gpt-3.5-turbo",
    "DE_1_3": "gpt-3.5-turbo",
    "DE_1_4": "gpt-3.5-turbo",
    "DE_1_5": "gpt-3.5-turbo",
    "DE_1_6": "gpt-3.5-turbo",
    "DE_1_7": "gpt-3.5-turbo",
    "DE_1_8": "gpt-3.5-turbo",
    "DE_2_1": "gpt-4-1106-preview",
    "DE_2_2": "gpt-4-1106-preview",
    "DE_2_3": "gpt-4-1106-preview",
    "DE_2_4": "gpt-4-1106-preview",
    "DE_2_5": "gpt-4-1106-preview",
    "DE_2_6": "gpt-4-1106-preview",
    "DE_2_7": "gpt-4-1106-preview",
    "DE_2_8": "gpt-4-1106-preview",
    "DE_3_1": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_2": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_3": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_4": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_5": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_6": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_7": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    "DE_3_8": 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3',
    }

# Dictionary to look up the original results of the experiments. key: experiment id, value: original result
DE_results_dict = {
    "DE_1_1": "A: 16%, B: 0%, C: 84%",
    "DE_1_2": "A: 68%, B: 0%, C: 32%",
    "DE_1_3": "A: 16%, B: 0%, C: 84%",
    "DE_1_4": "A: 68%, B: 0%, C: 32%",
    "DE_1_5": "A: 16%, B: 0%, C: 84%",
    "DE_1_6": "A: 68%, B: 0%, C: 32%",
    "DE_1_7": "A: 16%, B: 0%, C: 84%",
    "DE_1_8": "A: 68%, B: 0%, C: 32%",
    "DE_2_1": "A: 16%, B: 0%, C: 84%",
    "DE_2_2": "A: 68%, B: 0%, C: 32%",
    "DE_2_3": "A: 16%, B: 0%, C: 84%",
    "DE_2_4": "A: 68%, B: 0%, C: 32%",
    "DE_2_5": "A: 16%, B: 0%, C: 84%",
    "DE_2_6": "A: 68%, B: 0%, C: 32%",
    "DE_2_7": "A: 16%, B: 0%, C: 84%",
    "DE_2_8": "A: 68%, B: 0%, C: 32%",
    "DE_3_1": "A: 16%, B: 0%, C: 84%",
    "DE_3_2": "A: 68%, B: 0%, C: 32%",
    "DE_3_3": "A: 16%, B: 0%, C: 84%",
    "DE_3_4": "A: 68%, B: 0%, C: 32%",
    "DE_3_5": "A: 16%, B: 0%, C: 84%",
    "DE_3_6": "A: 68%, B: 0%, C: 32%",
    "DE_3_7": "A: 16%, B: 0%, C: 84%",
    "DE_3_8": "A: 68%, B: 0%, C: 32%",
}

# Dictionary to look up the scenario of each experiment. key: experiment id, value: scenario (1: With Decoy, 2: Without Decoy)
DE_scenario_dict = {
    "DE_1_1": 1,
    "DE_1_2": 2,
    "DE_1_3": 1,
    "DE_1_4": 2,
    "DE_1_5": 1,
    "DE_1_6": 2,
    "DE_1_7": 1,
    "DE_1_8": 2,
    "DE_2_1": 1,
    "DE_2_2": 2,
    "DE_2_3": 1,
    "DE_2_4": 2,
    "DE_2_5": 1,
    "DE_2_6": 2,
    "DE_2_7": 1,
    "DE_2_8": 2,
    "DE_3_1": 1,
    "DE_3_2": 2,
    "DE_3_3": 1,
    "DE_3_4": 2,
    "DE_3_5": 1,
    "DE_3_6": 2,
    "DE_3_7": 1,
    "DE_3_8": 2,
}

# Dictionary to look up, whether the experiment was primed or not. key: experiment id, value: priming (1: Primed, 0: Unprimed)
DE_priming_dict = {
    "DE_1_1": 0,
    "DE_1_2": 0,
    "DE_1_3": 1,
    "DE_1_4": 1,
    "DE_1_5": 0,
    "DE_1_6": 0,
    "DE_1_7": 1,
    "DE_1_8": 1,
    "DE_2_1": 0,
    "DE_2_2": 0,
    "DE_2_3": 1,
    "DE_2_4": 1,
    "DE_2_5": 0,
    "DE_2_6": 0,
    "DE_2_7": 1,
    "DE_2_8": 1,
    "DE_3_1": 0,
    "DE_3_2": 0,
    "DE_3_3": 1,
    "DE_3_4": 1,
    "DE_3_5": 0,
    "DE_3_6": 0,
    "DE_3_7": 1,
    "DE_3_8": 1,
}

# Dictionary to look up, whether answers were renamed and reordered or not. key: experiment id, value: indicator (1: Renamed and reordered, 0: Not renamed and reordered)
DE_reorder_dict = {
    "DE_1_1": 0,
    "DE_1_2": 0,
    "DE_1_3": 0,
    "DE_1_4": 0,
    "DE_1_5": 1,
    "DE_1_6": 1,
    "DE_1_7": 1,
    "DE_1_8": 1,
    "DE_2_1": 0,
    "DE_2_2": 0,
    "DE_2_3": 0,
    "DE_2_4": 0,
    "DE_2_5": 1,
    "DE_2_6": 1,
    "DE_2_7": 1,
    "DE_2_8": 1,
    "DE_3_1": 0,
    "DE_3_2": 0,
    "DE_3_3": 0,
    "DE_3_4": 0,
    "DE_3_5": 1,
    "DE_3_6": 1,
    "DE_3_7": 1,
    "DE_3_8": 1,
}


######################################## Experiment functions  ########################################

### Prospect Theory ###
# Function to run PT experiment with OpenAi models 
def PT_run_experiment_dashboard(experiment_id, n, temperature):

    """
    Function to query ChatGPT multiple times with a survey having answers designed as: A, B, C.
    
    Args:
        experiment_id (str): ID of the experiment to be run. Contains info about prompt and model
        n (int): Number of queries to be made
        temperature (int): Degree of randomness with range 0 (deterministic) to 2 (random)
        max_tokens (int): Maximum number of tokens in response object
        
    Returns:
        results (list): List containing count of answers for each option, also containing experiment_id, temperature and number of observations
        probs (list): List containing probability of each option being chosen, also containing experiment_id, temeperature and number of observations
    """
    
    answers = []
    for _ in range(n): 
        response = client.chat.completions.create(
            model = PT_model_dict[experiment_id], 
            max_tokens = 1,
            temperature = temperature, # range is 0 to 2
            messages = [
            {"role": "system", "content": "Only answer with the letter of the alternative you would choose without any reasoning."},        
            {"role": "user", "content": PT_experiment_prompts_dict[experiment_id]},
                   ])

        # Store the answer in the list
        answer = response.choices[0].message.content
        answers.append(answer.strip())
        # Update progress bar (given from either temperature loop, or set locally)
        #progress_bar.update(1)

    # Counting results
    A = answers.count("A")
    B = answers.count("B")
    C = answers.count("C")

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.DataFrame([experiment_id, temperature, A, B, C, len_correct, PT_model_dict[experiment_id], PT_scenario_dict[experiment_id], PT_priming_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Priming"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, PT_model_dict[experiment_id], PT_scenario_dict[experiment_id], PT_priming_dict[experiment_id]])
    probs = probs.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Priming"]))
    
    # Give out results
    return results, probs

# Function to run PT experiment with Meta's llama model
def PT_run_experiment_llama_dashboard(experiment_id, n, temperature):
    answers = []
    for _ in range(n):
        response = replicate.run(
            PT_model_dict[experiment_id],
            input = {
                "system_prompt": "Only answer with the letter of the alternative you would choose without any reasoning.",
                "temperature": temperature,
                "max_new_tokens": 2, 
                "prompt": PT_experiment_prompts_dict[experiment_id]
            }
        )
        # Grab answer and append to list
        answer = "" # Set to empty string, otherwise it would append the previous answer to the new one
        for item in response:
            answer = answer + item
        answers.append(answer.strip())

        # Update progress bar
        #progress_bar.update(1)

    # Counting results
    A = answers.count("A") # set to Q
    B = answers.count("B") # set to X
    C = answers.count("C") # set to Y

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.DataFrame([experiment_id, temperature, A, B, C, len_correct, PT_model_dict[experiment_id], PT_scenario_dict[experiment_id], PT_priming_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Priming"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, PT_model_dict[experiment_id], PT_scenario_dict[experiment_id], PT_priming_dict[experiment_id]])
    probs = probs.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Priming"]))
    
    # Give out results
    return results, probs


### Prospect Theory 2 ###
# Function to run PT2 experiment with OpenAI models
def PT2_run_experiment_dashboard(experiment_id, n, temperature):

    """
    Function to query ChatGPT multiple times with a survey having answers designed as: A, B, C.
    
    Args:
        experiment_id (str): ID of the experiment to be run. Contains info about prompt and model
        n (int): Number of queries to be made
        temperature (int): Degree of randomness with range 0 (deterministic) to 2 (random)
        max_tokens (int): Maximum number of tokens in response object
        
    Returns:
        results (list): List containing count of answers for each option, also containing experiment_id, temperature and number of observations
        probs (list): List containing probability of each option being chosen, also containing experiment_id, temeperature and number of observations
    """
    
    answers = []
    for _ in range(n): 
        response = client.chat.completions.create(
            model = PT2_model_dict[experiment_id], 
            max_tokens = 1,
            temperature = temperature, # range is 0 to 2
            messages = [
            {"role": "system", "content": "Only answer with the letter of the alternative you would choose without any reasoning."},        
            {"role": "user", "content": PT2_experiment_prompts_dict[experiment_id]},
                   ])

        # Store the answer in the list
        answer = response.choices[0].message.content
        answers.append(answer.strip())
        # Update progress bar (given from either temperature loop, or set locally)
        #progress_bar.update(1)

    # Counting results
    A = answers.count("A") 
    B = answers.count("B") 
    C = answers.count("C") 

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.DataFrame([experiment_id, temperature, A, B, C, len_correct, PT2_model_dict[experiment_id], PT2_scenario_dict[experiment_id], PT2_configuration_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Configuration"]))


    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, PT2_model_dict[experiment_id], PT2_scenario_dict[experiment_id], PT2_configuration_dict[experiment_id]])
    probs = results.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Configuration"]))
    
    # Give out results
    return results, probs

# Function to run PT2 experiment with Meta's LLama model
def PT2_run_experiment_llama_dashboard(experiment_id, n, temperature):
    answers = []
    for _ in range(n):
        response = replicate.run(
            PT2_model_dict[experiment_id],
            input = {
                "system_prompt": "Only answer with the letter of the alternative you would choose without any reasoning.",
                "temperature": temperature,
                "max_new_tokens": 2, 
                "prompt": PT2_experiment_prompts_dict[experiment_id]
            }
        )
        # Grab answer and append to list
        answer = "" # Set to empty string, otherwise it would append the previous answer to the new one
        for item in response:
            answer = answer + item
        answers.append(answer.strip())

        # Update progress bar
        #progress_bar.update(1)

    # Counting results
    A = answers.count("A") 
    B = answers.count("B") 
    C = answers.count("C") 

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.Dataframe([experiment_id, temperature, A, B, C, len_correct, PT2_model_dict[experiment_id], PT2_scenario_dict[experiment_id], PT2_configuration_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Configuration"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, PT2_model_dict[experiment_id], PT2_scenario_dict[experiment_id], PT2_configuration_dict[experiment_id]])
    probs = probs.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Configuration"]))
    
    # Give out results
    return results, probs


### Decoy Effect ###
# Function to count individual answers in DE experiment
def DE_count_answers(answers: list, experiment_id: str):
    if experiment_id in ["DE_1_1", "DE_1_3","DE_2_1", "DE_2_3", "DE_3_1", "DE_3_3"]:
        A = answers.count("A")
        B = answers.count("B")
        C = answers.count("C")
    elif experiment_id in ["DE_1_2", "DE_1_4", "DE_2_2", "DE_2_4", "DE_3_2", "DE_3_4"]:
        A = answers.count("A")
        B = 0 # Option B was removed
        C = answers.count("B") # makes comparison of results over prompts easier 
    elif experiment_id in ["DE_1_5", "DE_1_7", "DE_2_5", "DE_2_7", "DE_3_5", "DE_3_7"]:
        A = answers.count("Y")
        B = answers.count("Q")
        C = answers.count("X")
    elif experiment_id in ["DE_1_6", "DE_1_8", "DE_2_6", "DE_2_8", "DE_3_6", "DE_3_8"]:
        A = answers.count("Y")
        B = 0 # Option Q was removed
        C = answers.count("X")
    return A, B, C

# Function to count total correct answers in DE experiment 
def DE_correct_answers(answers: list, experiment_id: str):
    if experiment_id in ["DE_1_1", "DE_1_3","DE_2_1", "DE_2_3", "DE_3_1", "DE_3_3"]:
        len_correct = sum(1 for ans in answers if ans in ["A", "B", "C"])
    elif experiment_id in ["DE_1_2", "DE_1_4", "DE_2_2", "DE_2_4", "DE_3_2", "DE_3_4"]:
        len_correct = sum(1 for ans in answers if ans in ["A", "B"])
    elif experiment_id in ["DE_1_5", "DE_1_7", "DE_2_5", "DE_2_7", "DE_3_5", "DE_3_7"]:
        len_correct = sum(1 for ans in answers if ans in ["Y", "Q", "X"])
    elif experiment_id in ["DE_1_6", "DE_1_8", "DE_2_6", "DE_2_8", "DE_3_6", "DE_3_8"]:
        len_correct = sum(1 for ans in answers if ans in ["Y", "X"])
    return len_correct  

# Function to run DE experiment n times with OpenAI models 
def DE_run_experiment_dashboard(experiment_id: int, n: int, temperature: int):
    """
    Function to query ChatGPT multiple times with a survey having answers designed as: A, B, C.
    
    Args:
        experiment_id (str): ID of the experiment to be run. Contains info about prompt and model
        n (int): Number of queries to be made
        temperature (int): Degree of randomness with range 0 (deterministic) to 2 (random)
        max_tokens (int): Maximum number of tokens in response object
        
    Returns:
        results (list): List containing count of answers for each option, also containing experiment_id, temperature and number of observations
        probs (list): List containing probability of each option being chosen, also containing experiment_id, temeperature and number of observations
    """
    answers = []
    for _ in range(n): 
        response = client.chat.completions.create(
            model = DE_model_dict[experiment_id], 
            max_tokens = 5,
            temperature = temperature, # range is 0 to 2
            messages = [
            {"role": "system", "content": "Only answer with the letter of the alternative you would choose without any reasoning."},
            {"role": "user", "content": DE_experiment_prompts_dict[experiment_id]},
                   ])

        # Store the answer in the list
        answer = response.choices[0].message.content
        answers.append(answer.strip())
        # Update progress bar (given from either temperature loop, or set locally)
        #progress_bar.update(1)

    # Count the answers
    A, B, C = DE_count_answers(answers, experiment_id) # if/else statement of function deals with different answer options in different experiments
    
    # Count of correct answers
    len_correct = int(DE_correct_answers(answers, experiment_id)) # if/else of function makes sure that we count the correct answers according to the experiment id 

    # Collecting results in a list
    results = pd.DataFrame([experiment_id, temperature, A, B, C, len_correct, DE_model_dict[experiment_id], DE_scenario_dict[experiment_id], DE_priming_dict[experiment_id], DE_reorder_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Priming", "Reorder"]))

    # Calculate probabilities
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, DE_model_dict[experiment_id], DE_scenario_dict[experiment_id], DE_priming_dict[experiment_id], DE_reorder_dict[experiment_id]])
    probs = probs.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Priming", "Reorder"]))
    # Print progress
    # print(f"Experiment {experiment_id} with {n} observations, using {prompt_ids_dict[experiment_id]} and temperature {temperature} completed.")

    return results, probs 

# Function to run DE experiment n times with Meta's Llama model
def DE_run_experiment_llama_dashboard(experiment_id, n, temperature):
    answers = []
    for _ in range(n):
        response = replicate.run(
            DE_model_dict[experiment_id],
            input = {
                "system_prompt": "Only answer with the letter of the alternative you would choose without any reasoning.",
                "temperature": temperature,
                "max_new_tokens": 2, 
                "prompt": DE_experiment_prompts_dict[experiment_id]
            }
        )
        # Grab answer and append to list
        answer = "" # Set to empty string, otherwise it would append the previous answer to the new one
        for item in response:
            answer = answer + item
        answers.append(answer.strip())

        # Update progress bar
        #progress_bar.update(1)

    # Count the answers
    A, B, C = DE_count_answers(answers, experiment_id) # if/else statement of function deals with different answer options in different experiments
    
    # Count of correct answers
    len_correct = int(DE_correct_answers(answers, experiment_id)) # if/else of function makes sure that we count the correct answers according to the experiment id 

    # Collecting results in a list
    results = pd.DataFrame([experiment_id, temperature, A, B, C, len_correct, DE_model_dict[experiment_id], DE_scenario_dict[experiment_id], DE_priming_dict[experiment_id], DE_reorder_dict[experiment_id]])
    results = results.set_index(pd.Index(["Experiment", "Temp", "A", "B", "C", "Obs.", "Model", "Scenario", "Priming", "Reorder"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([experiment_id, temperature, p_a, p_b, p_c, len_correct, DE_model_dict[experiment_id], DE_scenario_dict[experiment_id], DE_priming_dict[experiment_id], DE_reorder_dict[experiment_id]])
    probs = probs.set_index(pd.Index(["Experiment", "Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model", "Scenario", "Priming", "Reorder"]))
    
    # Give out results
    return results, probs

# Initialize the app
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


### Individual Experiment ###
instructions = "Which alternative would you choose? Please answer by only giving the letter of the alternative you would choose without any reasoning."
def create_prompt2(prompt_design, answers):
    prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]} Option C: {answers[2]}. {instructions}"""
    return prompt

def create_prompt(prompt_design, answers):
    if len(answers) == 2:
        prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]}. {instructions}"""
    elif len(answers) == 3:
        prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]} Option C: {answers[2]}. {instructions}"""
    elif len(answers) == 4:
        prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]} Option C: {answers[2]} Option D: {answers[3]}. {instructions}"""
    elif len(answers) == 5:
        prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]} Option C: {answers[2]} Option D: {answers[3]} Option E: {answers[4]}. {instructions}"""
    elif len(answers) == 6:
        prompt = f"""{prompt_design} Option A: {answers[0]} Option B: {answers[1]} Option C: {answers[2]} Option D: {answers[3]} Option E: {answers[4]} Option F: {answers[5]}. {instructions}"""
    return prompt

def answer_randomization(options: list):
    # Generation of random letters 
    letters = random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', len(options))

    # Create a dictionary to map random letters to randomly ordered options
    options_random = random.sample(options, len(options))
    letter_mapping = {letters[i]: options_random[i] for i in range(len(options))}

    # Create the output string
    answer_options = ', '.join([f'{letters[i]}: {options_random[i]}' for i in range(len(options))])
    
    return answer_options

# Function to run individual experiment with OpenAI models
def run_individual_experiment_openai(prompt, model, iterations, temperature):
    model_answers = []
    for _ in range(iterations): 
        response = client.chat.completions.create(
            model = model, 
            max_tokens = 1,
            temperature = temperature, # range is 0 to 2
            messages = [
            {"role": "system", "content": "Only answer with the letter of the alternative you would choose without any reasoning."},        
            {"role": "user", "content": prompt},
                   ])

        # Store the answer in the list
        answer = response.choices[0].message.content
        model_answers.append(answer.strip())

    # Counting results
    A = model_answers.count("A")
    B = model_answers.count("B")
    C = model_answers.count("C")

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in model_answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.DataFrame([temperature, A, B, C, len_correct, model])
    results = results.set_index(pd.Index(["Temp", "A", "B", "C", "Obs.", "Model"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100 
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([temperature, p_a, p_b, p_c, len_correct, model])
    probs = probs.set_index(pd.Index(["Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model"]))
    
    # Give out results
    return results, probs

# Function to run individual experiment with Meta's llama model
def run_individual_experiment_llama(prompt, model, iterations, temperature):
    model_answers = []
    for _ in range(iterations):
        response = replicate.run(
            model,
            input = {
                "system_prompt": "Only answer with the letter of the alternative you would choose without any reasoning.",
                "temperature": temperature,
                "max_new_tokens": 2, 
                "prompt": prompt
            }
        )
        # Grab answer and append to list
        answer = "" # Set to empty string, otherwise it would append the previous answer to the new one
        for item in response:
            answer = answer + item
        model_answers.append(answer.strip())

    # Counting results
    A = model_answers.count("A") 
    B = model_answers.count("B") 
    C = model_answers.count("C") 

    # Count of "correct" answers, sums over indicator function thack checks if answer is either A, B or C
    len_correct = sum(1 for ans in model_answers if ans in ["A", "B", "C"])

    # Collecting results in a list
    results = pd.DataFrame([temperature, A, B, C, len_correct, model])
    results = results.set_index(pd.Index(["Temp", "A", "B", "C", "Obs.", "Model"]))

    # Getting percentage each answer
    p_a = (A / (len_correct + 0.000000001)) * 100
    p_b = (B / (len_correct + 0.000000001)) * 100
    p_c = (C / (len_correct + 0.000000001)) * 100

    # Collect probabilities in a dataframe
    probs = pd.DataFrame([temperature, p_a, p_b, p_c, len_correct, model])
    probs = probs.set_index(pd.Index(["Temp", "p(A)", "p(B)", "p(C)", "Obs.", "Model"]))
    
    # Give out results
    return results, probs





# Optics of sidebar
SIDEBAR_STYLE = {
    "position": "fixed", # remains in place when scrolling
    "top": 0, # begins at top of page
    "left": 0, # begins at left of page
    "bottom": 0, # ends at bottom of page
    "width": "16rem", # obvious, rem is "unit" of indentation
    "padding": "1.5rem 1.5rem", # distance of sidebar entries from top and left
    "background-color": "#c8f7f3",
}

# Optics of main page content
CONTENT_STYLE = {
    "margin-left": "18rem", # indentation of main content from left side (sidebar is 16rem wide)
    "margin-right": "2rem", # indentation of main content from right side
    "padding": "2rem 2rem", # distance of main content from top and bottom
}

# Create the sidebar
sidebar = html.Div(
    [
        html.H2("Navigation", className="display-6"),
        html.Hr(),
        html.P(
            "Feel free to explore", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Overview", href = "/experiments/overview"),
                dbc.DropdownMenuItem("Decoy Effect", href="/experiments/decoy-effect"),
                dbc.DropdownMenuItem("Prospect Theory", href="/experiments/prospect-theory"),
                dbc.DropdownMenuItem("Sunk Cost Fallacy", href="/experiments/sunk-cost"),
                dbc.DropdownMenuItem("Ultimatum Game", href="/experiments/ultimatum"),
                dbc.DropdownMenuItem("Loss Aversion", href="/experiments/loss-aversion"),
            ],
            label = "Experiments",
            nav = True,
        ),
                dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Overview", href = "/live-experiment/overview"),
                dbc.DropdownMenuItem("Prospect Theory", href = "/live-experiment/prospect-theory"),
                dbc.DropdownMenuItem("Prospect Theory 2.0", href = "/live-experiment/prospect-theory-2"),
                dbc.DropdownMenuItem("Decoy Effect", href = "/live-experiment/decoy-effect"),
                dbc.DropdownMenuItem("Sunk Cost Fallacy", href = "live-experiment/sunk-cost"),
                dbc.DropdownMenuItem("Ultimatum Game", href = "/live-experiment/ultimatum"),
                dbc.DropdownMenuItem("Loss Aversion", href = "/live-experiment/loss-aversion"),
                dbc.DropdownMenuItem("Individual Experiment", href = "/live-experiment/individual"),
            ],       
            label = "Live Experiment",
            nav = True,
        ),
                dbc.NavLink("Chatbot", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

#########################################  Experiment Page Designs  #########################################
# Add content for pages
# Start Page
start_page = [
            html.H1("Do Large Language Models Behave like a Human?", className="page-heading"),
            html.P("""Large Language models hold huge potential for a wide range of applications either for private, but also for professional use. 
                   One possible question that is of especially interesting for market research, is whether these models behave human-like enough to be used as surrogates for 
                   human participants in experiments. This dashboard is a first attempt to answer this question."""),
            html.P("Feel free to explore more pages using the navigation menu.")]

# Decoy Page
decoy_page = [
    html.H1("Decoy Effect Experiment", className="page-heading"), 
    html.Hr(),
    html.P(["""The decoy effect describes a phenomenon, in which  consumers preferences between two products change, once a third option is added. This third option is designed 
            to be asymmetrically dominated, meaning that it is entirely inferior to one of the previous options, but only partially inferior to the other. Once this asymetrically 
            dominated option, the Decoy, is present, more people will now tend to choose the dominating option than before. A decoy product can therefore be used to influence consumer's
            decision making and increase saless of a specific product merely through the presence of an additional alternative.""",
            html.Br(),
            html.Br(),
            """Our experiment aims to recreate the findings of Ariely in his 2008 book *Predictably Irrational*. There, he asked 100 students from MIT's Sloan School of Management 
            to choose between the following options:""",
            html.Br(),
            html.Br(),
            "A: One-year subscription to Economist.com. Includes online access to all articles from The Economist since 1997, priced at 59$.",
            html.Br(),
            "B: One-year subscription to the print edition of The Economist, priced at 125$.",
            html.Br(),
            "C: One-year subscription to the print edition of The Economist and online access to all articles from The Economist since 1997, priced at 125$.",
            html.Br(),
            html.Br(),
            "In this example, option B serves as the decoy option.",
            html.Br(), 
            "When presented with ", html.B("all three options"), " Ariely found, that ", html.B("84%"), " of the participants chose option ", html.B("C"), " while only ", html.B("16%"), " chose option ", html.B("A"),".",
            html.Br(),
            "However, once ", html.B("option B was removed"), " and the choice had to be made only between A and C, ", html.B("68%"), " of the participants chose option ", html.B("A"), " while only ", html.B("32%"), " chose option ", html.B("C"),".",
            html.Br(),
            html.Br(),
            """In the experiments below, we examine how various Large Language Models react to this kind of experiment. We therefore queried 3 different models over a range of possible 
            temperature values using either primed or unprimed prompts. On top of that, we investigated to what extent the models' responses change, when we rename and reorder the 
            answer options. In the case of primed prompts, we instructed the model to be a market researcher, who knows about the Decoy Effect in product pricing."""]),
            html.Br(),
            html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[
                    dcc.Dropdown(
                         id = "decoy-scenario-dropdown",
                         options = [
                              {"label": "Scenario 1: All options present", "value": 1},
                              {"label": "Scenario 2: Decoy option removed", "value": 2},
                         ],
                         value = 1,
                         style={'width': '75%'},
                    ),
                    dcc.Dropdown(
                         id = "decoy-priming-dropdown",
                         options = [
                              {"label": "Unprimed prompt", "value": 0},
                              {"label": "Primed prompt", "value": 1},
                            ],
                            value = 0,
                            style={'width': '75%'},
                    ),
                    dcc.Dropdown(
                         id = "decoy-reordering-dropdown",
                         options = [
                              {"label": "Original order", "value": 0},
                              {"label": "Answer options reordered", "value": 1},
                            ],
                            value = 0,
                            style={'width': '75%'},
                    ),
                    dcc.Dropdown(
                         id = "decoy-model-dropdown",
                         options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                              {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                              {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style={'width': '75%'},
                    ),
                         ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
            dcc.Graph(id="decoy-plot-output", style={'width': '70%', 'height': '60vh'}),
        ],
        
        style={'display': 'flex', 'flexDirection': 'row'})]

# Prospect Page
prospect_page = [
     html.H1("Prospect Theory and Mental Accounting Experiment", className="page-heading"),
     html.Hr(),
     html.P(["""According to Prospect Theory and Mental Accounting, financial gains and losses are booked into different fictitious accounts. On top of that, 
            relative to a reference point, losses weigh more heavily than gains and the perceived sum of two individual gains/losses will, in absolute terms, be larger than 
            one single gain/loss of the same amount. In the context of Marketing, four main rules can be derived by this theory:""",
            html.Br(),
            html.Br(),
            "1) Segregation of gains",
            html.Br(),
            "2) Integration of losses",
            html.Br(), 
            "3) Cancellation of losses against larger gains",
            html.Br(),
            "4) Segregation of silver linings",
            html.Br(),
            html.Br(),
            """One possible practical implication each of these rules hold, is each reflected in the different scenarios we examine below.""",
            html.Br(),
            """In order to research how Large Language models react to this kind of experiment, we queried multiple models over different temperature values and used either primed 
            or unprimed prompts. The results of our experiments are visualized below. The original results are taken
            from Thaler, Richard (1985), “Mental Accounting and Consumer Choice,” Marketing Science, 4 (3), 199–214 and the prompts we query the Language Models with are
            constructed so that we can stay as close to the original phrasing as possible, while still instructing the models sufficiently well to produce meaningful results.
            For every scenario, the participants could decide on either Mister A, Mister B or a No-difference option.
            In the case of primed experiments, we instructed the model to be a market researcher, that knows about Prospect Theory and Mental Accounting.""",
            html.Br(),
            html.Br(),
            ]),

html.H2("Experiment 1: Recreating the original study"),
html.Br(),
# Scenario 1: Segregation of gains
html.Div(
    children=[
        html.H3("Scenario 1: Segregation of gains"),
        html.Hr(),
        html.Div(
            children=[
                html.P(
                    [   html.Br(),
                        html.Br(),
                        "The original phrasing, used in the experiment by Thaler, is as follows:",
                        html.Br(),
                        "Mr. A was given tickets to lotteries involving the World Series. He won $50 in one lottery and $25 in the other.",
                        html.Br(),
                        "Mr. B was given a ticket to a single, larger World Series lottery. He won $75. Who was happier?",
                        html.Br(),
                        html.Br(),
                        "A: Mister A",
                        html.Br(),
                        "B: Mister B",
                        html.Br(),
                        "C: No difference",
                    ]
                ),
                html.Img(src=PT_og_scenario1, style={'max-width': '100%', 'max-height': '300px', 'margin-left': '60px', 'margin-top': '20px'}),


            ],
            style={'display': 'flex', 'flexDirection': 'row'},
        ),
html.Div(
    children=[
        html.Div(
            children=[
                html.H5("Select experiment design:"),
                dcc.RadioItems(
                    id="prospect-scenario1-radio1",
                    options=[
                        {"label": "Unprimed", "value": 0},
                        {"label": "Primed", "value": 1},
                    ],
                    value=0,
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
                dcc.RadioItems(
                    id="prospect-scenario1-radio2",
                    options=[
                        {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                        {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                        {"label": "LLama-2-70b", "value": "llama-2-70b"},
                    ],
                    value="gpt-3.5-turbo",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
            ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
        ),
        dcc.Graph(id="prospect-plot1", style={'width': '70%', 'height': '60vh'}),
    ],
    style={'display': 'flex', 'flexDirection': 'row'},
)]),

    
# Scenario 2: Integration of losses
html.Div(
    children=[
        html.H3("Scenario 2: Integration of losses"),
        html.Hr(),
        html.Div(
            children=[
                html.P(
                    [   html.Br(),
                        html.Br(),
                        "The original phrasing, used in the experiment by Thaler, is as follows:",
                        html.Br(),
                        """Mr. A received a letter from the IRS saying that he made a minor arithmetical mistake on his
                        tax return and owed $100. He received a similar letter the same day from his state income tax
                        authority saying he owed $50. There were no other repercussions from either mistake.""",
                        html.Br(),
                        """Mr. B received a letter from the IRS saying that he made a minor arithmetical mistake on his tax
                        return and owed $150. There were no other repercussions from his mistake. Who was more upset?""",
                        html.Br(),
                        html.Br(),
                        "A: Mister A",
                        html.Br(),
                        "B: Mister B",
                        html.Br(),
                        "C: No difference",
                    ]
                ),
                html.Img(src=PT_og_scenario2, style={'max-width': '100%', 'max-height': '300px', 'margin-left': '60px', 'margin-top': '20px'}),


            ],
            style={'display': 'flex', 'flexDirection': 'row'},
        ),
html.Div(
    children=[
        html.Div(
            children=[
                html.H5("Select experiment design:"),
                dcc.RadioItems(
                    id="prospect-scenario2-radio1",
                    options=[
                        {"label": "Unprimed", "value": 0},
                        {"label": "Primed", "value": 1},
                    ],
                    value=0,
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
                dcc.RadioItems(
                    id="prospect-scenario2-radio2",
                    options=[
                        {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                        {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                        {"label": "LLama-2-70b", "value": "llama-2-70b"},
                    ],
                    value="gpt-3.5-turbo",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
            ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
        ),
        dcc.Graph(id="prospect-plot2", style={'width': '70%', 'height': '60vh'}),
    ],
    style={'display': 'flex', 'flexDirection': 'row'},
)]),

    
# Scenario 3: Cancellation of losses against larger gains
html.Div(
    children=[
        html.H3("Scenario 3: Cancellation of losses against larger gains"),
        html.Hr(),
        html.Div(
            children=[
                html.P(
                    [   html.Br(),
                        html.Br(),
                        "The original phrasing, used in the experiment by Thaler, is as follows:",
                        html.Br(),
                        """Mr. A bought his first New York State lottery ticket and won $100. Also, in a freak accident,
                        he damaged the rug in his apartment and had to pay the landlord $80.""",
                        html.Br(),
                        "Mr. B bought his first New York State lottery ticket and won $20. Who was happier",
                        html.Br(),
                        html.Br(),
                        "A: Mister A",
                        html.Br(),
                        "B: Mister B",
                        html.Br(),
                        "C: No difference",
                    ]
                ),
                html.Img(src=PT_og_scenario3, style={'max-width': '100%', 'max-height': '300px', 'margin-left': '60px', 'margin-top': '20px'}),


            ],
            style={'display': 'flex', 'flexDirection': 'row'},
        ),
html.Div(
    children=[
        html.Div(
            children=[
                html.H5("Select experiment design:"),
                dcc.RadioItems(
                    id="prospect-scenario3-radio1",
                    options=[
                        {"label": "Unprimed", "value": 0},
                        {"label": "Primed", "value": 1},
                    ],
                    value=0,
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
                dcc.RadioItems(
                    id="prospect-scenario3-radio2",
                    options=[
                        {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                        {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                        {"label": "LLama-2-70b", "value": "llama-2-70b"},
                    ],
                    value="gpt-3.5-turbo",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
            ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
        ),
        dcc.Graph(id="prospect-plot3", style={'width': '70%', 'height': '60vh'}),
    ],
    style={'display': 'flex', 'flexDirection': 'row'},
)]),

# Scenario 4: Segregation of silver linings
html.Div(
    children=[
        html.H3("Scenario 4: Segregation of silver linings"),
        html.Hr(),
        html.Div(
            children=[
                html.P(
                    [   html.Br(),
                        html.Br(),
                        "The original phrasing, used in the experiment by Thaler, is as follows:",
                        html.Br(),
                        """Mr. A's car was damaged in a parking lot. He had to spend $200 to repair the damage. 
                        The same day the car was damaged, he won $25 in the office football pool.""",
                        html.Br(),
                        "Mr. B's car was damaged in a parking lot. He had to spend $175 to repairthe damage. Who was more upset?",
                        html.Br(),
                        html.Br(),
                        "A: Mister A",
                        html.Br(),
                        "B: Mister B",
                        html.Br(),
                        "C: No difference",
                    ]
                ),
                html.Img(src=PT_og_scenario4, style={'max-width': '100%', 'max-height': '300px', 'margin-left': '60px', 'margin-top': '20px'}),


            ],
            style={'display': 'flex', 'flexDirection': 'row'},
        ),
html.Div(
    children=[
        html.Div(
            children=[
                html.H5("Select experiment design:"),
                dcc.RadioItems(
                    id="prospect-scenario4-radio1",
                    options=[
                        {"label": "Unprimed", "value": 0},
                        {"label": "Primed", "value": 1},
                    ],
                    value=0,
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
                dcc.RadioItems(
                    id="prospect-scenario4-radio2",
                    options=[
                        {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                        {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                        {"label": "LLama-2-70b", "value": "llama-2-70b"},
                    ],
                    value="gpt-3.5-turbo",
                    inputStyle={"margin-right": "10px"},
                    labelStyle={
                        "display": "inline-block",
                        "margin-right": "20px",
                    },
                ),
            ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
        ),
        dcc.Graph(id="prospect-plot4", style={'width': '70%', 'height': '60vh'}),
    ],
    style={'display': 'flex', 'flexDirection': 'row'},
)]),
html.Br(),
html.Hr(),


## Experiment 2
html.H2("Experiment 2: Odd numbers and unfair scenarios"),
html.Hr(),
html.P(["""The Prospect Theory value function explains why individuals tend to assess the perceived value of e.g. a sum of multiple gains as larger, 
        than one individual sum of the same amount. Since Large Language Models are trained on human data, including for example customer reviews on sales platforms,
        they might reflect these patterns.""",
        html.Br(), 
        """But how do LLMs react, if in the given scenarios, one individual is financially clearly better off than the other? And what if we did not deal with small,
        even numbers, but rather large and odd ones?""",
        html.Br(),
        "Another ", html.B("key concept of prospect theory is decreasing sensitivity"),":", 
        " A loss of 50$ subtracted from a total amount of 1000$ will not hurt as much, as if we initially only had 100$, hence losing 50% of our total possession.", 
        html.Br(),
        html.Br(),
        "In order to research these 2 aspects, we created 6 configurations for every scenario (1-4):",
        html.Br(),
        html.Br(),
        "- Configuration 1: Original numbers scaled by factor Pi * 100",
        html.Br(),
        "- Configuration 2: Original numbers scaled by factor Pi * 42",
        html.Br(),
        "- Configuration 3: A is better off by 25$",
        html.Br(),
        "- Configuration 4: A is better off by 50$",
        html.Br(),
        "- Configuration 5: B is better off by 25$",
        html.Br(),
        "- Configuration 6: B is better off by 50$",
        html.Br()]),
    html.Div(
        children = [
            html.Div(
                children = [
                    html.H5("Select experiment design:", style = {'margin-left': '-75px'}),
                    dcc.Dropdown(
                        id = "prospect2-scenario-dropdown",
                        options = [
                            {"label": "Scenario 1: Segregation of gains", "value": 1},
                            {"label": "Scenario 2: Integration of losses", "value": 2},
                            {"label": "Scenario 3: Cancellation of losses against larger gains", "value": 3},
                            {"label": "Scenario 4: Segregation of silver linings", "value": 4},
                        ],
                        value = 1,
                        style = {'width': '75%'},
                    ),
                    dcc.Dropdown(
                         id = "prospect2-configuration-dropdown",
                            options = [
                                {"label": "Configuration 1: Odd numbers 1", "value": 1},
                                {"label": "Configuration 2: Odd numbers 2", "value": 2},
                                {"label": "Configuration 3: A is better off by 25$", "value": 3},
                                {"label": "Configuration 4: A is better off by 50$", "value": 4},
                                {"label": "Configuration 5: B is better off by 25$", "value": 5},
                                {"label": "Configuration 6: B is better off by 50$", "value": 6},
                                ],
                                value = 1,
                                style = {'width': '75%'},
                    ),
                    dcc.Dropdown(
                         id = "prospect2-model-dropdown",
                         options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                                {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                                {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style = {'width': '75%'},
                    )],                 
    
                style = {'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
            dcc.Graph(id = "prospect2-plot", style={'width': '70%', 'height': '60vh'}),
        ],
        style={'display': 'flex', 'flexDirection': 'row'})]


# Sunk Cost Fallacy Page
sunk_cost_page = [
    html.H1("Sunk Cost Fallacy", className="page-heading"),
    html.P('Description of how the experiments were conducted: ...'),
    
    # Experiment 1
    html.H3("Experiment 1"),
    html.P(id='experiment-1-prompt'),

    html.Div(
        style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'},
        children=[
            html.Div(
                style={'width': '25%', 'margin-right': '30px', 'align-self': 'flex-start', 'margin-top': '170px'}, 
                children=[
                    html.H6('Temperature Value'),
                    dcc.Slider(
                        id="Temperature_1",
                        min=0.5,
                        max=1.5,
                        step=0.5,
                        marks={0.5: '0.5', 1: '1', 1.5: '1.5'},
                        value=1,
                    ),
                    
                    html.H6('Amount of Sunk Cost (Cost of Theater Performance)', style={'margin-top': '50px'}), 
                    dcc.Dropdown(
                        id="Sunk-Cost",
                        options=[
                            {'label': '$90', 'value': 90},
                            {'label': '$250', 'value': 250},
                            {'label': '$10,000', 'value': 10_000}
                        ],
                        value=90,
                        style={'width': '100%'}
                    ),
                ]
            ),
            
            dcc.Graph(id="sunk-cost-plot-1-output", style={'width': '65%', 'height': '70vh'}),
        ]
    ),
    
    # Experiment 2
    html.H3("Experiment 2"),
    html.P(["""Suppose you bought a case of good Bordeaux in the futures \
            market for $20 a bottle. The wine now sells at auction for about $75. \
                You have decided to drink a bottle. Which of the following best captures \
                    your feeling of the cost to you of drinking the bottle?"""
    ]),
    html.P('(Same answer options, but in different order):'),
    html.Div([
        html.Div([
            html.H6('Answer Option Order 1', style={'margin-top': '15px'}),
            html.P(["A: $0. I already paid for it.",
                    html.Br(),  # Line break
                    "B: $20, what I paid for.",
                    html.Br(),  # Line break
                    "C: $20, plus interest.",
                    html.Br(),  # Line break
                    "D: $75, what I could get if I sold the bottle.",
                    html.Br(),  # Line break
                    "E: -$55, I get to drink a bottle that is worth $75 that I only paid \
                        $20 for so I save money by drinking the bottle."]),
        ], style={'width': '40%', 'display': 'inline-block', 'margin-bottom': '60px', 'vertical-align': 'top'}),

        html.Div([
            html.H6('Answer Option Order 2', style={'margin-top': '15px'}),
            html.P(["A: $75",
                    html.Br(),  # Line break
                    "B: -$55",
                    html.Br(),  # Line break
                    "C: $0",
                    html.Br(),  # Line break
                    "D: $20",
                    html.Br(),  # Line break
                    "E: $20, plus interest"]),
        ], style={'width': '20%', 'display': 'inline-block', 'margin-bottom': '60px', 'vertical-align': 'top'}),

        html.Div([
            html.H6('Answer Option Order 3', style={'margin-top': '15px'}),
            html.P(["A: -$55",
                    html.Br(),  # Line break
                    "B: $75",
                    html.Br(),  # Line break
                    "C: $20 plus interest",
                    html.Br(),  # Line break
                    "D: $0",
                    html.Br(),  # Line break
                    "E: $20"]),
        ], style={'width': '20%', 'display': 'inline-block', 'margin-bottom': '60px', 'vertical-align': 'top'}),
    ]),
        

    html.Div(
        style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'},
        children=[
            html.Div(
                style={'width': '25%', 'margin-right': '30px', 'align-self': 'flex-start', 'margin-top': '170px'},  
                children=[
                    html.H6('Temperature Value'),
                    dcc.Slider(
                        id="Temperature_2",
                        min=0.5,
                        max=1.5,
                        step=0.5,
                        marks={0.5: '0.5', 1: '1', 1.5: '1.5'},
                        value=1,
                    ),
                    
                    html.H6('Model', style={'margin-top': '50px'}),  
                    dcc.Dropdown(
                        id="Model",
                        options=[
                            {'label': 'gpt-3.5-turbo-1106', 'value': 'gpt-3.5-turbo-1106'},
                            {'label': 'gpt-4-1106-preview', 'value': 'gpt-4-1106-preview'},
                            {'label': 'llama-2-70b', 'value': 'llama-2-70b'},
                        ],
                        value='gpt-3.5-turbo-1106',
                        style={'width': '100%'}  
                    ),
                ]
            ),
            
            dcc.Graph(id="sunk-cost-plot-2-output", style={'width': '65%', 'height': '70vh'}),  # Adjust height as needed
        ]
    ),
]


# Loss Aversion Page
loss_aversion_page = [
    html.H1("Loss Aversion", className="page-heading"),
    html.P('Description of how the experiment was conducted: ...'),
    
    html.Div([
        # Experiment 1
        html.Div([
            html.H6("Scenario with gains:"),
            html.P(["You are offered two choices. Which choice would you prefer?",
                    html.Br(),  # Line break
                    html.Br(),  # Line break
                    "A: A sure gain of $100.",
                    html.Br(),  # Line break
                    "B: A 50% chance to gain $200 and a 50% chance to lose $0."
            ]),
        ], style={'width': '40%', 'display': 'inline-block', 'margin-bottom': '60px'}),

        html.Div([
            html.H6("Scenario with losses:"),
            html.P(["You are offered two choices. Which choice would you prefer?",
                    html.Br(),  # Line break
                    html.Br(),  # Line break
                    "A: A sure loss of $100.",
                    html.Br(),  # Line break
                    "B: A 50% chance to lose $200 and a 50% chance to lose $0."
            ]),
        ], style={'width': '40%', 'display': 'inline-block', 'margin-bottom': '60px'})
    ]),

    html.Div(
        style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'},
        children=[
            html.Div(
                style={'width': '25%', 'margin-right': '30px', 'align-self': 'center'}, 
                children=[
                    html.H6('Temperature Value'),
                    dcc.Slider(
                        id="Temperature",
                        min=0.5,
                        max=1.5,
                        step=0.5,
                        marks={0.5: '0.5', 1: '1', 1.5: '1.5'},
                        value=1,
                    ),
                ]
            ),
            
            dcc.Graph(id="loss_aversion_plot_output", style={'width': '65%', 'height': '70vh'}),
        ]
    )
]


# Individual Prospect Theory Page
individual_prospect_page = [
    html.H1("Prospect Theory Live Experiment", className="page-heading"), 
    html.Hr(),
    html.P("""Choose an experiment configuration from the options below and run the experiment yourself. You can choose 4 different scenarios, 3 different models and 
           primed vs. unprimed prompts."""),
    html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[
                            html.Label("Select a scenario", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect-live-scenario-dropdown",
                            options = [
                              {"label": "Scenario 1: Segregation of gains", "value": 1},
                              {"label": "Scenario 2: Integration of losses", "value": 2},
                              {"label": "Scenario 3: Cancellation of losses against larger gains", "value": 3},
                              {"label": "Scenario 4: Segregation of silver linings", "value": 4},
                         ],
                         value = 1,
                         style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select a language model", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect-live-model-dropdown",
                            options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                              {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                              {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select Prompt design", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect-live-priming-dropdown",
                            options = [
                              {"label": "Unprimed", "value": 0},
                              {"label": "Primed", "value": 1},
                            ],
                            value = 0,
                            style={'width': '75%', 'margin': 'auto'},

                    ),     
                            html.Label("Select number of requests", style={'textAlign': 'center'}),                
                            dbc.Input(
                            id = "prospect-live-iterations", 
                            type = "number",
                            value = 1, 
                            min = 0, 
                            max = 100, 
                            step = 1,
                            style={'width': '57%', 'margin': 'auto'}, # apparently default width for input is different from dropdown
                    ),      
                    html.Div(
                        [
                            html.Label("Select Temperature value"),             
                            dcc.Slider(
                                id="prospect-live-temperature",
                                min=0.01,
                                max=2,
                                step=0.01,
                                marks={0.01: '0.01', 2: '2'},
                                value=0.8,
                                tooltip={'placement': 'top'},
                            ),
                        ],
                    ),

                    # Add a button to trigger calback
                    html.Button('Run the experiment', id = 'prospect-live-update-button', n_clicks = None),
                         ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
        dcc.Graph(id="prospect-live-output", style={'width': '70%', 'height': '60vh'}),
        ],
    style={'display': 'flex', 'flexDirection': 'row'}
    ),
    # Additional text section
    html.Div(
            id='prospect-live-experiment-design',
            style={'textAlign': 'center', 'margin': '20px'},
    ),
    html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
]



# Individual Prospect Theory Page 2
individual_prospect_page2 = [
    html.H1("Prospect Theory Live Experiment 2.0", className="page-heading"), 
    html.Hr(),
    html.P("""Choose an experiment configuration from the options below and run the experiment yourself. You can choose 4 different scenarios, 3 different models and 
           6 different configurations."""),
    html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[
                            html.Label("Select a scenario", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect2-live-scenario-dropdown",
                            options = [
                              {"label": "Scenario 1: Segregation of gains", "value": 1},
                              {"label": "Scenario 2: Integration of losses", "value": 2},
                              {"label": "Scenario 3: Cancellation of losses against larger gains", "value": 3},
                              {"label": "Scenario 4: Segregation of silver linings", "value": 4},
                         ],
                         value = 1,
                         style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select a language model", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect2-live-model-dropdown",
                            options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                              {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                              {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select a configuration", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "prospect2-live-configuration-dropdown",
                            options = [
                                {"label": "Configuration 1: Odd numbers 1", "value": 1},
                                {"label": "Configuration 2: Odd numbers 2", "value": 2},
                                {"label": "Configuration 3: A is better off by 25$", "value": 3},
                                {"label": "Configuration 4: A is better off by 50$", "value": 4},
                                {"label": "Configuration 5: B is better off by 25$", "value": 5},
                                {"label": "Configuration 6: B is better off by 50$", "value": 6},
                                ],
                                value = 1,
                                style = {'width': '75%', 'margin': 'auto'},
                    ),   
                            html.Label("Select number of requests", style={'textAlign': 'center'}),                
                            dbc.Input(
                            id = "prospect-live-iterations", 
                            type = "number",
                            value = 0, 
                            min = 0, 
                            max = 100, 
                            step = 1,
                            style={'width': '57%', 'margin': 'auto'}, # apparently default width for input is different from dropdown
                    ),      
                    html.Div(
                        [
                            html.Label("Select Temperature value"),             
                            dcc.Slider(
                                id="prospect2-live-temperature",
                                min=0.01,
                                max=2,
                                step=0.01,
                                marks={0.01: '0.01', 2: '2'},
                                value=0.8,
                                tooltip={'placement': 'top'},
                            ),
                        ],
                    ),

                    # Add a button to trigger calback
                    html.Button('Run the experiment', id = 'prospect2-live-update-button', n_clicks = None),
                         ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
        dcc.Graph(id="prospect2-live-output", style={'width': '70%', 'height': '60vh'}),
        ],
    style={'display': 'flex', 'flexDirection': 'row'}
    ),
    # Additional text section
    html.Div(
            id='prospect2-live-experiment-design',
            style={'textAlign': 'center', 'margin': '20px'},
    ),
    html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
]



# Individual Decoy Effect page
individual_decoy_page = [
    html.H1("Decoy Effect Live Experiment", className="page-heading"), 
    html.Hr(),
    html.P("""Choose an experiment configuration from the options below and run the experiment yourself. You can choose 2 different scenarios, 3 different models,
           primed vs. unprimed prompts and reordered vs original answer options."""),
    html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[
                            html.Label("Select a scenario", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "decoy-live-scenario-dropdown",
                            options = [
                              {"label": "Scenario 1: All answer options", "value": 1},
                              {"label": "Scenario 2: Decoy option removed", "value": 2},
                         ],
                         value = 1,
                         style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select a language model", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "decoy-live-model-dropdown",
                            options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                              {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                              {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style={'width': '75%', 'margin': 'auto'},
                    ),
                            html.Label("Select Prompt design", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "decoy-live-priming-dropdown",
                            options = [
                              {"label": "Unprimed", "value": 0},
                              {"label": "Primed", "value": 1},
                            ],
                            value = 0,
                            style={'width': '75%', 'margin': 'auto'},

                    ),     
                            html.Label("Select answer structure", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "decoy-live-reorder-dropdown",
                            options = [
                              {"label": "Original order", "value": 0},
                              {"label": "Renamed & reordered", "value": 1},
                            ],
                            value = 0,
                            style={'width': '75%', 'margin': 'auto'},

                    ),   
                            html.Label("Select number of requests", style={'textAlign': 'center'}),                
                            dbc.Input(
                            id = "decoy-live-iterations", 
                            type = "number",
                            value = 0, 
                            min = 0, 
                            max = 100, 
                            step = 1,
                            style={'width': '57%', 'margin': 'auto'}, # apparently default width for input is different from dropdown
                    ),      
                    html.Div(
                        [
                            html.Label("Select Temperature value"),             
                            dcc.Slider(
                                id="decoy-live-temperature",
                                min=0.01,
                                max=2,
                                step=0.01,
                                marks={0.01: '0.01', 2: '2'},
                                value=0.8,
                                tooltip={'placement': 'top'},
                            ),
                        ],
                    ),

                    # Add a button to trigger calback
                    html.Button('Run the experiment', id = 'decoy-live-update-button', n_clicks = None),
                         ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
        dcc.Graph(id="decoy-live-output", style={'width': '70%', 'height': '60vh'}),
        ],
    style={'display': 'flex', 'flexDirection': 'row'}
    ),
    # Additional text section
    html.Div(
            id='decoy-live-experiment-design',
            style={'textAlign': 'center', 'margin': '20px'},
    ),
    html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
]


# Page for actual individual experiment
individual_experiment_page = [
    html.H1("Conduct your own individual experiment", className="page-heading"), 
    html.Hr(),
    html.P("""Think of multiple-choice-style experiment to conduct. Choose a prompt, a model, and answer options to run the experiment yourself."""),
    html.Br(),
    html.Div(
        children=[
            html.Div(
                children=[
                            html.Label("Select a scenario", style={'textAlign': 'center'}),
                            dcc.Textarea(
                            id='individual-prompt',
                            value="You are a random pedestrian being chosen for a survey. The question is: Would you rather:",
                            style={'width': '100%', 'height': 300},
                             ),
                            # Answer option A 
                            html.Label("Answer option A", style={'textAlign': 'center'}),
                            dcc.Textarea(
                            id='individual-answer-a',
                            value='Win 50$',
                            style={'width': '100%', 'height': 300},
                             ),
                            # Answer option B
                            html.Label("Answer option B", style={'textAlign': 'center'}),
                            dcc.Textarea(
                            id='individual-answer-b',
                            value='Lose 50$',
                            style={'width': '100%', 'height': 300},
                             ),   
                            # Answer option C   
                            html.Label("Answer option C", style={'textAlign': 'center'}),
                            dcc.Textarea(
                            id='individual-answer-c',
                            value='Win 100$',
                            style={'width': '100%', 'height': 300},
                             ),                                            
                            html.Label("Select a language model", style={'textAlign': 'center'}),
                            dcc.Dropdown(
                            id = "individual-model-dropdown",
                            options = [
                              {"label": "GPT-3.5-Turbo", "value": "gpt-3.5-turbo"},
                              {"label": "GPT-4-1106-Preview", "value": "gpt-4-1106-preview"},
                              {"label": "LLama-2-70b", "value": "llama-2-70b"},
                            ],
                            value = "gpt-3.5-turbo",
                            style={'width': '75%', 'margin': 'auto'},
                    ),

                            html.Label("Select number of requests", style={'textAlign': 'center'}),                
                            dbc.Input(
                            id = "individual-iterations", 
                            type = "number",
                            value = 0, 
                            min = 0, 
                            max = 100, 
                            step = 1,
                            style={'width': '57%', 'margin': 'auto'}, # apparently default width for input is different from dropdown
                    ),      
                    html.Div(
                        [
                            html.Label("Select Temperature value"),             
                            dcc.Slider(
                                id="individual-temperature",
                                min=0.01,
                                max=2,
                                step=0.01,
                                marks={0.01: '0.01', 2: '2'},
                                value=0.8,
                                tooltip={'placement': 'top'},
                            ),
                        ],
                    ),

                    # Add a button to trigger calback
                    html.Button('Run the experiment', id = 'individual-update-button', n_clicks = None),
                         ],
            style={'display': 'flex', 'flexDirection': 'column', 'align-items': 'center', 'width': '50%', 'align-self': 'center'},
            ),
        dcc.Graph(id="individual-output", style={'width': '70%', 'height': '60vh'}),
        ],
    style={'display': 'flex', 'flexDirection': 'row'}
    ),
    # Additional text section
    html.Div(
            id='individual-experiment-design',
            style={'textAlign': 'center', 'margin': '20px'},
    ),
    html.Button("Download CSV", id="btn_csv"),
        dcc.Download(id="download-dataframe-csv"),
]
################################################## Callbacks ##################################################

### Callback for prospect page

## Experiment 1
# Scenario 1
@app.callback(
     Output("prospect-plot1", "figure"),
     [Input("prospect-scenario1-radio1", "value"), # priming
        Input("prospect-scenario1-radio2", "value")] # model

)
def update_prospect_plot1(selected_priming, selected_model):
        return plot_results(model = selected_model, priming = selected_priming, df = PT_probs, scenario = 1) 

# Scenario 2
@app.callback(
        Output("prospect-plot2", "figure"),
        [Input("prospect-scenario2-radio1", "value"), # priming
        Input("prospect-scenario2-radio2", "value")] #  model

)
def update_prospect_plot2(selected_priming, selected_model):
        return plot_results(model = selected_model, priming = selected_priming, df = PT_probs, scenario = 2) 


# Scenario 3
@app.callback(
        Output("prospect-plot3", "figure"),
        [Input("prospect-scenario3-radio1", "value"), # priming
        Input("prospect-scenario3-radio2", "value")] # model
)  
def update_prospect_plot3(selected_priming, selected_model):    
        return plot_results(model = selected_model, priming = selected_priming, df = PT_probs, scenario = 3)
    
# Scenario 4
@app.callback(
        Output("prospect-plot4", "figure"),
        [Input("prospect-scenario4-radio1", "value"), #  priming
        Input("prospect-scenario4-radio2", "value")] #  model
)
def update_prospect_plot4(selected_priming, selected_model):
        return plot_results(model = selected_model, priming = selected_priming, df = PT_probs, scenario = 4)

## Experiment 2
@app.callback(
     Output("prospect2-plot", "figure"),
     [Input("prospect2-scenario-dropdown", "value"),
      Input("prospect2-configuration-dropdown", "value"),
      Input("prospect2-model-dropdown", "value")]
)
def update_prospect2_plot(selected_scenario, selected_configuration, selected_model):
    df = PT2_probs[PT2_probs["Configuration"] == selected_configuration]
    return plot_results(model = selected_model, df = df, scenario = selected_scenario, priming = 0) # all experiments are unprimed


# Callback for decoy page
@app.callback(
    Output("decoy-plot-output", "figure"),
    [Input("decoy-scenario-dropdown", "value"),
     Input("decoy-priming-dropdown", "value"),
     Input("decoy-reordering-dropdown", "value"),
     Input("decoy-model-dropdown", "value")]
     )
def update_decoy_plot(selected_scenario, selected_priming, selected_reordering, selected_model):
    # Pre-select dataframe (plot_results disregards reordering option)
    df = DE_probs[DE_probs["Reorder"] == selected_reordering]
    return plot_results(scenario = selected_scenario, priming = selected_priming, model = selected_model, df = df)
    
    
# Callback for Sunk Cost Fallacy Experiment 1
@app.callback(
    [Output("sunk-cost-plot-1-output", "figure"),
     Output("experiment-1-prompt", "children")],
    [Input("Temperature_1", "value"),
     Input("Sunk-Cost", "value")]
)
def update_sunk_cost_plot_1(selected_temperature, selected_sunk_cost):
    figure = plot_sunk_cost_1(selected_temperature, selected_sunk_cost)
    
    # Update the description of Experiment 1
    experiment_description = [
        f"""Assume that you have spent ${selected_sunk_cost} for a ticket to a theater performance. \
        Several weeks later you buy a $30 ticket to a rock concert. You think you will \
        enjoy the rock concert more than the theater performance. As you are putting your \
        just-purchased rock concert ticket in your wallet, you notice that both events \
        are scheduled for the same evening. The tickets are non-transferable, nor \
        can they be exchanged. You can use only one of the tickets and not the other. \
        Which ticket will you use? """,
        html.Br(),  # Line break
        html.Br(),  # Line break
        "A: Theater performance.",
        html.Br(),  # Line break
        "B: Rock concert."
    ]

    return figure, experiment_description
    
# Callback for Sunk Cost Fallacy Experiment 2
@app.callback(
    Output("sunk-cost-plot-2-output", "figure"),
    [Input("Temperature_2", "value"),
     Input("Model", "value")]
)
def update_sunk_cost_plot_2(selected_temperature, selected_model):
    return plot_sunk_cost_2(selected_temperature, selected_model)


# Callback for Loss Aversion Experiment
@app.callback(
    Output("loss_aversion_plot_output", "figure"),
    [Input("Temperature", "value")]
)
def update_loss_averion_plot(selected_temperature):
    return plot_loss_aversion(selected_temperature)

#  Callback for Individual Prospect Theory Experiment
@app.callback(
    [Output("prospect-live-output", "figure"),
     Output('prospect-live-experiment-design', 'children')], 
    [Input("prospect-live-update-button", "n_clicks")],
    [State("prospect-live-scenario-dropdown", "value"),
     State("prospect-live-model-dropdown", "value"),
     State("prospect-live-priming-dropdown", "value"),
     State("prospect-live-iterations", "value"),
     State("prospect-live-temperature", "value")]
     )
def update_prospect_live_plot(n_clicks, selected_scenario, selected_model, selected_priming, selected_iterations, selected_temperature):
    # Check if the button was clicked
    if n_clicks is not None:
        if selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 0:
            experiment_id = "PT_1_1"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 0:
            experiment_id = "PT_1_2"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_priming == 0:
            experiment_id = "PT_1_3"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_priming == 0:
            experiment_id = "PT_1_4"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 1:
            experiment_id = "PT_1_5"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 1:
            experiment_id = "PT_1_6"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_priming == 1:
            experiment_id = "PT_1_7"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_priming == 1:
            experiment_id = "PT_1_8"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 0:
            experiment_id = "PT_2_1"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 0:
            experiment_id = "PT_2_2"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_priming == 0:
             experiment_id = "PT_2_3"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_priming == 0:
            experiment_id = "PT_2_4"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 1:
            experiment_id = "PT_2_5"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 1:
            experiment_id = "PT_2_6"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_priming == 1:
            experiment_id = "PT_2_7"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_priming == 1:
            experiment_id = "PT_2_8"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 0:
            experiment_id = "PT_3_1"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 0:
            experiment_id = "PT_3_2"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_priming == 0:
            experiment_id = "PT_3_3"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_priming == 0:
            experiment_id = "PT_3_4"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 1:
            experiment_id = "PT_3_5"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 1:
            experiment_id = "PT_3_6"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_priming == 1:
            experiment_id = "PT_3_7"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_priming == 1:
            experiment_id = "PT_3_8"
        else:
            experiment_id = None
        
        # Run Experiment for selected parameters
        if selected_model == "llama-2-70b":
            results, probs = PT_run_experiment_llama_dashboard(experiment_id, selected_iterations, selected_temperature)
        else:
            results, probs = PT_run_experiment_dashboard(experiment_id, selected_iterations, selected_temperature)
        n_clicks = None
        prompt = html.P(f"The prompt used in this experiment is: {PT_experiment_prompts_dict[experiment_id]} The original results were: {PT_results_dict[experiment_id]}.")
        return plot_results_individual_recreate(probs), prompt

# Callback for Individual Prospect Theory Experiment 2.0
@app.callback(
    [Output("prospect2-live-output", "figure"),
     Output('prospect2-live-experiment-design', 'children')],
    [Input("prospect2-live-update-button", "n_clicks")],
    [State("prospect2-live-scenario-dropdown", "value"),
     State("prospect2-live-model-dropdown", "value"),
     State("prospect2-live-configuration-dropdown", "value"),
     State("prospect-live-iterations", "value"),
     State("prospect2-live-temperature", "value")]
     )
def update_prospect2_live_plot(n_clicks, selected_scenario, selected_model, selected_configuration, selected_iterations, selected_temperature):
    # Check if button was clicked
    if n_clicks is not None:
        if selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 1:
            experiment_id = "PT2_1_1_1"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 2:
            experiment_id = "PT2_1_1_2"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 3:
            experiment_id = "PT2_1_1_3"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 4:
            experiment_id = "PT2_1_1_4"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 5:
            experiment_id = "PT2_1_1_5"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_configuration == 6:
            experiment_id = "PT2_1_1_6"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 1:
            experiment_id = "PT2_2_1_1"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 2:
            experiment_id = "PT2_2_1_2"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 3:
            experiment_id = "PT2_2_1_3"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 4:
            experiment_id = "PT2_2_1_4"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 5:
            experiment_id = "PT2_2_1_5"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_configuration == 6:
            experiment_id = "PT2_2_1_6"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 1:
            experiment_id = "PT2_3_1_1"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 2:
            experiment_id = "PT2_3_1_2"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 3:
            experiment_id = "PT2_3_1_3"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 4:
            experiment_id = "PT2_3_1_4"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 5:
            experiment_id = "PT2_3_1_5"
        elif selected_scenario == 3 and selected_model == "gpt-3.5-turbo" and selected_configuration == 6:
            experiment_id = "PT2_3_1_6"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 1:
            experiment_id = "PT2_4_1_1"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 2:
            experiment_id = "PT2_4_1_2"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 3:
            experiment_id = "PT2_4_1_3"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 4:
            experiment_id = "PT2_4_1_4"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 5:
            experiment_id = "PT2_4_1_5"
        elif selected_scenario == 4 and selected_model == "gpt-3.5-turbo" and selected_configuration == 6:
            experiment_id = "PT2_4_1_6"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 1:
            experiment_id = "PT2_1_2_1"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 2:
            experiment_id = "PT2_1_2_2"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 3:
            experiment_id = "PT2_1_2_3"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 4:
            experiment_id = "PT2_1_2_4"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 5:
            experiment_id = "PT2_1_2_5"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_configuration == 6:
            experiment_id = "PT2_1_2_6"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 1:
            experiment_id = "PT2_2_2_1"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 2:
            experiment_id = "PT2_2_2_2"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 3:
            experiment_id = "PT2_2_2_3"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 4:
            experiment_id = "PT2_2_2_4"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 5:
            experiment_id = "PT2_2_2_5"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_configuration == 6:
            experiment_id = "PT2_2_2_6"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 1:
            experiment_id = "PT2_3_2_1"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 2:
            experiment_id = "PT2_3_2_2"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 3:
            experiment_id = "PT2_3_2_3"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 4:
            experiment_id = "PT2_3_2_4"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 5:
            experiment_id = "PT2_3_2_5"
        elif selected_scenario == 3 and selected_model == "gpt-4-1106-preview" and selected_configuration == 6:
            experiment_id = "PT2_3_2_6"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 1:
            experiment_id = "PT2_4_2_1"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 2:
            experiment_id = "PT2_4_2_2"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 3:
            experiment_id = "PT2_4_2_3"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 4:
            experiment_id = "PT2_4_2_4"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 5:
            experiment_id = "PT2_4_2_5"
        elif selected_scenario == 4 and selected_model == "gpt-4-1106-preview" and selected_configuration == 6:
            experiment_id = "PT2_4_2_6"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 1:
            experiment_id = "PT2_1_3_1"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 2:
            experiment_id = "PT2_1_3_2"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 3:
            experiment_id = "PT2_1_3_3"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 4:    
            experiment_id = "PT2_1_3_4"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 5:
            experiment_id = "PT2_1_3_5"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_configuration == 6:
            experiment_id = "PT2_1_3_6"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 1:
            experiment_id = "PT2_2_3_1"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 2:
            experiment_id = "PT2_2_3_2"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 3:
            experiment_id = "PT2_2_3_3"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 4:
            experiment_id = "PT2_2_3_4"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 5:
            experiment_id = "PT2_2_3_5"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_configuration == 6:
            experiment_id = "PT2_2_3_6"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 1:
            experiment_id = "PT2_3_3_1"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 2:
            experiment_id = "PT2_3_3_2"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 3:
            experiment_id = "PT2_3_3_3"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 4:
            experiment_id = "PT2_3_3_4"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 5:
            experiment_id = "PT2_3_3_5"
        elif selected_scenario == 3 and selected_model == "llama-2-70b" and selected_configuration == 6:
            experiment_id = "PT2_3_3_6"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 1:
            experiment_id = "PT2_4_3_1"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 2:
            experiment_id = "PT2_4_3_2"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 3:
            experiment_id = "PT2_4_3_3"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 4:
            experiment_id = "PT2_4_3_4"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 5:
            experiment_id = "PT2_4_3_5"
        elif selected_scenario == 4 and selected_model == "llama-2-70b" and selected_configuration == 6:
            experiment_id = "PT2_4_3_6"
        else:
            experiment_id = None

        # Run Experiment for selected parameters
        if selected_model == "llama-2-70b":
            results, probs = PT2_run_experiment_llama_dashboard(experiment_id, selected_iterations, selected_temperature)
        else:
            results, probs = PT2_run_experiment_dashboard(experiment_id, selected_iterations, selected_temperature)
        n_clicks = None
        prompt = html.P(f"The prompt used in this experiment is: {PT2_experiment_prompts_dict[experiment_id]} The original results were: {PT2_results_dict[experiment_id]}.")
        return plot_results_individual_recreate(probs), prompt

# Callback for individual Decoy Effect experiment 
@app.callback(
    [Output("decoy-live-output", "figure"),
     Output('decoy-live-experiment-design', 'children')],
    [Input("decoy-live-update-button", "n_clicks")],
    [State("decoy-live-scenario-dropdown", "value"),
     State("decoy-live-model-dropdown", "value"),
     State("decoy-live-priming-dropdown", "value"),
     State("decoy-live-reorder-dropdown", "value"),
     State("decoy-live-iterations", "value"),
     State("decoy-live-temperature", "value")]
     )

def update_decoy_live_plot(n_clicks, selected_scenario, selected_model, selected_priming, selected_reordering, selected_iterations, selected_temperature):
    # Check if button was clicked
    if n_clicks is not None:
        if selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_1_1"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_1_2"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_1_3"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_1_4"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_1_5"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_1_6"
        elif selected_scenario == 1 and selected_model == "gpt-3.5-turbo" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_1_7"
        elif selected_scenario == 2 and selected_model == "gpt-3.5-turbo" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_1_8"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_2_1"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_2_2"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_2_3"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_2_4"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_2_5"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_2_6"
        elif selected_scenario == 1 and selected_model == "gpt-4-1106-preview" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_2_7"
        elif selected_scenario == 2 and selected_model == "gpt-4-1106-preview" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_2_8"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_3_1"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 0 and selected_reordering == 0:
            experiment_id = "DE_3_2"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_3_3"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 1 and selected_reordering == 0:
            experiment_id = "DE_3_4"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_3_5"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 0 and selected_reordering == 1:
            experiment_id = "DE_3_6"
        elif selected_scenario == 1 and selected_model == "llama-2-70b" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_3_7"
        elif selected_scenario == 2 and selected_model == "llama-2-70b" and selected_priming == 1 and selected_reordering == 1:
            experiment_id = "DE_3_8"
        else:
            experiment_id = "test"
        
        # Run Experiment for selected parameters
        if selected_model == "llama-2-70b":
            results, probs = DE_run_experiment_llama_dashboard(experiment_id, selected_iterations, selected_temperature)
        else:
            results, probs = DE_run_experiment_dashboard(experiment_id, selected_iterations, selected_temperature)
        n_clicks = None
        prompt = html.P(f"The prompt used in this experiment is: {DE_experiment_prompts_dict[experiment_id]} The original results were: {DE_results_dict[experiment_id]}.")
        return plot_results_individual_recreate(probs), prompt

# Callback for individual live experiment
@app.callback(
    [Output("individual-output", "figure"),
     Output('individual-experiment-design', 'children')],
    [Input("individual-update-button", "n_clicks")],
    [State("individual-prompt", "value"),
     State("individual-answer-a", "value"),
     State("individual-answer-b", "value"),
     State("individual-answer-c", "value"),
     State("individual-model-dropdown", "value"),
     State("individual-iterations", "value"),
     State("individual-temperature", "value")]
     )

def update_individual_experiment(n_clicks, prompt, answer_a, answer_b, answer_c, selected_model, selected_iterations, selected_temperature):
    # Check if button was clicked
    if n_clicks is not None:
        answers = [answer_a, answer_b, answer_c]
        prompt = create_prompt2(prompt, answers)
        print(f"Prompt: {prompt}")
        print(f" Selected model: {selected_model}")
        print(f"Selected iterations: {selected_iterations}")
        print(f"Selected temperature: {selected_temperature}")
        if selected_model == "llama-2-70b":
            selected_model = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
            results, probs = run_individual_experiment_llama(prompt, selected_model, selected_iterations, selected_temperature)
        else:
            results, probs = run_individual_experiment_openai(prompt, selected_model, selected_iterations, selected_temperature)
        n_clicks = None
        prompt = html.P(f"The prompt used in this experiment is: {prompt}")
        return plot_results_individual(probs), prompt             
        # Run Experiment for selected parameters

# Callback for navigation bar
@app.callback(Output("page-content", "children"),
             [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.P(start_page)
    elif pathname == "/page-1":
        return html.P("Experiments are not yet implemented. Sorry!")
    elif pathname == "/experiments/overview":
        return html.P("Overview of experiments is not yet implemented. Sorry!")
    elif pathname == "/experiments/decoy-effect":
        return html.P(decoy_page)
    elif pathname == "/experiments/prospect-theory":
        return html.P(prospect_page)
    elif pathname == "/experiments/sunk-cost":
        return html.P(sunk_cost_page)
    elif pathname == "/experiments/ultimatum":
        return html.P("Ultimatum experiment is not yet implemented. Sorry!")
    elif pathname == "/experiments/loss-aversion":
        return html.P(loss_aversion_page)
    elif pathname == "/live-experiment/overview":
        return html.P("Overview of live experiments is not yet implemented. Sorry!")
    elif pathname == "/live-experiment/prospect-theory":
         return html.P(individual_prospect_page)
    elif pathname == "/live-experiment/prospect-theory-2":
        return html.P(individual_prospect_page2)
    elif pathname == "/live-experiment/decoy-effect":
         return html.P(individual_decoy_page)
    elif pathname == "/live-experiment/sunk-cost":
            return html.P("Sunk cost live experiment not yet implemented. Sorry!")
    elif pathname == "/live-experiment/ultimatum":
         return html.P("Individual ultimatum experiment is not yet implemented. Sorry!")
    elif pathname == "/live-experiment/loss-aversion":
         return html.P("Loss aversion live experiment not yet implemented. Sorry!")
    elif pathname == "/live-experiment/individual":
            return html.P(individual_experiment_page)
    elif pathname == "/page-3":
        return html.P("This chatbot is not yet implemented. Sorry!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )



if __name__ == "__main__":
    app.run_server(port=8888, debug = False)