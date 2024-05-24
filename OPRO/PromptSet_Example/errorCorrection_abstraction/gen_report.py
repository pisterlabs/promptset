import statistics
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
    
    step_to_accuracy = {
        step: (
            statistics.mean(opro_results[step].values()),
            max(opro_results[step].values()),
            min(opro_results[step].values()),
        )
        for step in opro_results
    }

    # Plot step to accuracy as a continuous line graph, including stdevs as highlighted error bars
    fig, ax = plt.subplots()
    ax.errorbar(
        step_to_accuracy.keys(),
        [accuracy[0] for accuracy in step_to_accuracy.values()],
        yerr=[
            [accuracy[0] - accuracy[2] for accuracy in step_to_accuracy.values()],  # Lower errors
            [accuracy[1] - accuracy[0] for accuracy in step_to_accuracy.values()]   # Upper errors
        ],
        fmt="o",
    )

    ax.set_xticks([step for step in step_to_accuracy.keys() if int(step) % 5 == 0])
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Step to Accuracy")

    # Add a pink point to the plot
    ax.plot(0, opro_results["0"][initial_prompt], 'o', color='pink')

    # Add legend to show pink dot is initial prompt accuracy, blue dot is avg prompt accuracy
    ax.legend(["Initial Prompt Accuracy", "Average Prompt Accuracy"], loc="lower right")


    # save the plot
    plt.savefig(f"{GRAPH_DIR_PATH}/{ID}.png")

    # return initial prompt accuracy and best prompt accuracy
    return opro_results["0"][initial_prompt], max([max(v.values()) for v in opro_results.values()])


# load cache file
with open(CACHE_FILE) as f:
    prompt_scores = json.load(f)

id_to_promptStats = {}
for p, p_score in prompt_scores.items():
    ID = p_score["ID"]
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
df.to_html("opro_stats.html", escape=False)
# webbrowser.open("opro_stats.html")