import openai
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def get_gpt_response(description,text,model_id):
  openai.api_key = None
  response = openai.ChatCompletion.create(
    model=model_id,
    messages=[
      {
        "role": "system",
        "content": f"{description}"
      },
      {
        "role": "user",
        "content": f"{text}"
      }
    ],
    max_tokens=4000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  return response.choices[0].message["content"]

def inference_main(description,text,model_id):
    openai.api_key = ""
    df=pd.DataFrame()
    df['quarter']=0
    
    text_parts = [
    text[:len(text) // 4],
    text[len(text) // 4:2 * len(text) // 4],
    text[2 * len(text) // 4:3 * len(text) // 4],
    text[3 * len(text) // 4:],
    ]

    # Initialize a list to store the GPT responses
    gpt_responses = []

    # Iterate through the text parts and get responses
    for part in text_parts:
        response = get_gpt_response(description,part,model_id)
        gpt_responses.append(response)

 
    for i, chunk in enumerate(gpt_responses):
        parts = []
        current_part = {"category": None, "Text": ""}
        lines = chunk.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(("P:", "S:", "O:")):
                if current_part["category"] is not None:
                    parts.append(current_part.copy())
                current_part["category"] = line.split(":")[0]
                current_part["Text"] = line.split(":", 1)[1].strip()
            else:
                current_part["Text"] += "\n" + line

        parts.append(current_part)  # Add the last part
        chunk_df = pd.DataFrame(parts)
        chunk_df['quarter'] = i
        df = df.append(chunk_df, ignore_index=True)

    return df

def visualize():
    df1 = pd.read_excel('/mnt/d/Users/BKU/SashaBehrouzi/Documents/DPA/Data/output_gpt4.xlsx')
    df2 = pd.read_excel('/mnt/d/Users/BKU/SashaBehrouzi/Documents/DPA/Data/output_finetune275_try_1.xlsx')
    baseline_ps_list=[91,24,18,27]
    # Group the DataFrames by 'quarter' and 'category', then count the occurrences
    grouped1 = df1.groupby(['quarter', 'category']).size().unstack(fill_value=0)
    grouped2 = df2.groupby(['quarter', 'category']).size().unstack(fill_value=0)

    # Calculate the ratio of 'P' to 'S' counts in percentage for both DataFrames
    grouped1['P_to_S_Ratio'] = (grouped1['P'] / grouped1['S']) * 100
    grouped2['P_to_S_Ratio'] = (grouped2['P'] / grouped2['S']) * 100

    # Create a DataFrame with fixed values for each quarter
    fixed_values = pd.DataFrame({
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
        'fixed_value': baseline_ps_list
        })

    # Find the common y-axis limits (starting from 0)
    y_min = 0
    y_max = max(grouped1['P_to_S_Ratio'].max(), grouped2['P_to_S_Ratio'].max(), fixed_values['fixed_value'].max())

    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(40, 20), sharey=True)  # sharey=True shares the y-axis

    # Set the global style for all subplots
    sns.set(style="whitegrid")

    # Rearrange the order of subplots
    axes = [axes[0], axes[2], axes[1]]  # Change the order here

    # Loop through each subplot and plot the data
    for i, (ax, data, title) in enumerate(zip(axes, [fixed_values['fixed_value'], grouped1['P_to_S_Ratio'], grouped2['P_to_S_Ratio']], ['Baseline', 'gpt3.5 fine-tuned', 'gpt 4'])):  # Change the order here
        sns.lineplot(data=data, marker='o', markersize=10, alpha=0.5, ax=ax)
        ax.set_title(f'P-S Ratio Over Quarters {title}', fontsize=14)
        ax.set_xlabel('Quarters', fontsize=12)
        ax.set_ylabel('P-S Ratio (%)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])

        # Set the common y-axis limits
        ax.set_ylim(y_min, y_max)

        # Add labels at the start of each data point
        for x, y in enumerate(data):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(0, 10), ha='center')

        # Adjust layout
    plt.tight_layout()

    # Return the Matplotlib figure instead of displaying it
    return fig


def inference(description,text,model_id):
    data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'San Francisco', 'Los Angeles']}

    df = pd.DataFrame(data)
    return df
