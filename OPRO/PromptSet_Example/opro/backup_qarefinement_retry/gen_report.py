import statistics
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import json, os
import pandas as pd
import webbrowser


CACHE_FILE = "testingSetScores.json"
GRAPH_DIR_PATH = "graphs"

def make_graph(ID, initial_prompt):
    # If graph dir doesn't exist, create it
    if not os.path.exists(GRAPH_DIR_PATH):
        os.makedirs(GRAPH_DIR_PATH)

    SAVE_PATH = f"{ID}/training_results.json"
    # load opro.json
    with open(SAVE_PATH) as f:
        opro_results = json.load(f)
    
    step_to_score = {
        step: (
            statistics.mean(opro_results[step].values()),
            max(opro_results[step].values()),
            min(opro_results[step].values()),
        )
        for step in opro_results if step != "0"  # Exclude step 0
    }

    # Create a gridspec to hold the main plot and the sidebar
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 20]) 

    # Plot step to score as a continuous line graph, including stdevs as highlighted error bars
    ax = plt.subplot(gs[1])
    ax.errorbar(
        step_to_score.keys(),
        [score[0] for score in step_to_score.values()],
        yerr=[
            [score[0] - score[2] for score in step_to_score.values()],  # Lower errors
            [score[1] - score[0] for score in step_to_score.values()]   # Upper errors
        ],
        fmt="o",
    )
    ax.hlines(
        y=opro_results["0"][initial_prompt],
        xmin=0,
        xmax=int(max(step_to_score.keys())) if step_to_score else 1,
        colors='pink',
        linestyles='dashed',
    )

    ax.set_xticks([step for step in step_to_score.keys()])
    ax.set_xlabel("Step")
    ax.set_title("Step vs. Score")

    # Add legend to show pink dot is initial prompt score, blue dot is avg prompt score
    ax.legend(["Initial Prompt Score", "Average Prompt Score"], loc="lower right")

    # Add a sidebar for x=0
    ax1 = plt.subplot(gs[0])
    step_to_score = {
        step: (
            statistics.mean(opro_results[step].values()),
            max(opro_results[step].values()),
            min(opro_results[step].values()),
        )
        for step in opro_results if step == "0"  # Exclude step 0
    }
    ax1.errorbar(
        step_to_score.keys(),
        [score[0] for score in step_to_score.values()],
        yerr=[
            [score[0] - score[2] for score in step_to_score.values()],  # Lower errors
            [score[1] - score[0] for score in step_to_score.values()]   # Upper errors
        ],
        fmt="o",
    )

    initial_prompt_val = opro_results["0"][initial_prompt]
    ax1.plot(0, initial_prompt_val, 'o', color='pink')  # Add the pink point to the sidebar
    ax1.set_xticks([0])
    ax1.set_yticks(range(0, 101, 10))
    ax1.set_ylabel("Score")
    ax1.set_xlabel('Seed Prompts\n(Step 0)')
    # ax1.set_title(" " * 20)

    # save the plot
    plt.savefig(f"{GRAPH_DIR_PATH}/{ID}.png")

    # return initial prompt score and best prompt score
    return opro_results["0"][initial_prompt], max([max(v.values()) for v in opro_results.values()])


# load cache file
with open(CACHE_FILE) as f:
    prompt_scores = json.load(f)

id_to_promptStats = {}
for p, p_score in prompt_scores.items():
    ID = p_score["ID"]
    category = p_score["category"]
    initial_prompt_score = p_score["initial_prompt"]
    optimized_prompt_score = p_score["optimized_prompt"]
    initial_prompt = next(iter(initial_prompt_score))
    initial_test_score = initial_prompt_score[initial_prompt]
    optimized_prompt = next(iter(optimized_prompt_score))
    optimized_test_score = optimized_prompt_score[optimized_prompt]
    initial_train_score, optimized_train_score = make_graph(ID, initial_prompt)
    id_to_promptStats[ID] = {
        "graph": f'<img src="{GRAPH_DIR_PATH}/{ID}.png" alt="img">',
        # "graph": f'![graph]({GRAPH_DIR_PATH}/{ID}.png)',
        "initial_prompt": initial_prompt,
        "category": category,
        "initial_train_score": initial_train_score,
        "initial_test_score": initial_test_score,
        "optimized_prompt": optimized_prompt,
        "optimized_train_score": optimized_train_score,
        "optimized_test_score": optimized_test_score,
    }


    print(f"Prompt ID: {ID}")
    print(f"Initial prompt score: {initial_prompt_score}")
    print(f"Optimized prompt score: {optimized_prompt_score}")

df = pd.DataFrame.from_dict(id_to_promptStats, orient='index')
df.to_html("opro_stats.html", escape=False, index=False)
# webbrowser.open("opro_stats.html")