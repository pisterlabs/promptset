
import openai
import csv
import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from .run_llm_code import run_llm_code

def get_prompt(task, list_codeblocks_generated, list_performance_pipelines, generated_pipelines=1):
    string_list_code_and_score = [
        f"Pipeline: \n {list_codeblocks_generated[i]}, \n Score/error: \n {list_performance_pipelines[i]}" for i in
        range(len(list_codeblocks_generated))]
    string_list_code_and_score = '\n'.join(string_list_code_and_score)
    if task == 'classification':
        metric_prompt = 'Log loss'
    else:
        metric_prompt = 'Mean Squared Error'
    return f"""
I have some sklearn pipelines with their corresponding scores/errors, where higher scores indicate better performance. The task is ‘{task}’.

"
{string_list_code_and_score}
"

Your new task is to create a new pipeline inspired by previous examples. Your goal is to improve the performance of the models

Code formatting for all the pipelines created:
```python
# Import all the packages necesaries, always call 'make_pipeline' from sklearn (essentialy copy all the packages from previous example)
# Create the pipeline considering the preprocessing steps from the previous examples (since they are indispensable), mainly if there is a 'columntransformer'
# Tune hyperparameters of the main model for the new pipeline 'pipe'. It must be different from previous examples
# Call the 'fit' function to feed 'pipe' with 'X_train' and 'y_train'
```end

Each codeblock generates exactly {generated_pipelines} different and useful pipelines, which will be evaluated with "{metric_prompt}". 
Each codeblock ends with "```end" and starts with "```python". 

Codeblock:

"""

def optimize_LLM(
        X,
        y,
        model="gpt-3.5-turbo",
        display_method="markdown",
        task='classification',
        iterations_max=8,
        identifier='',
):
    list_pipelines_optimized = []
    list_score_optimized = []

    def custom_convert(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    def generate_code(messages):
        if model == "skip":
            return ""
        if model =='gpt-4':
            max_tokens = 6000
        else:
            max_tokens = 1000

        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stop=["```end"],
            temperature=1,
            max_tokens=max_tokens, #change to 6000 when using gpt-4
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    def execute_and_evaluate_code_block(code):
        if task == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,  stratify=y, random_state=0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        try:
            pipe = run_llm_code(
                code,
                X_train,
                y_train,
            )
            performance = pipe.score(X_test, y_test)

        except Exception as e:
            pipe = None
            display_method(f"Error in code execution. {type(e)} {e}")
            display_method(f"```python\n{format_for_display(code)}\n```\n")
            return e, None, None

        return None, performance, pipe

    iterations = 0
    while iterations <= iterations_max:
        track_pipelines = pd.read_csv(f'pipelines_{identifier}.csv', names=['Timestamp', 'Pipeline', 'Score'])
        track_pipelines['Score'] = track_pipelines['Score'].apply(custom_convert)
        # Convert the 'Score' column to numeric, errors='coerce' converts non-float values to NaN
        track_pipelines['Score'] = pd.to_numeric(track_pipelines['Score'], errors='coerce')
        track_pipelines = track_pipelines.dropna(subset=['Score'])
        track_pipelines = track_pipelines.sort_values('Score', ascending=False)  # ascending order
        track_pipelines = track_pipelines.head(3) # Only 3 models
        list_codeblocks_generated = list(track_pipelines['Pipeline'])
        list_performance_pipelines = list(track_pipelines['Score'])
        prompt = get_prompt(task, list_codeblocks_generated, list_performance_pipelines)

        messages = [
            {
                "role": "system",
                "content": "You are an expert datascientist assistant creating new Pipelines for a dataset X_train, y_train (all pipelines must be called only 'pipe'). You answer only by generating code. Take a deep breath and work on this problem step-by-step.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            time.sleep(60) # Wait 1 minute before next request
            continue
        e, performance, pipe = execute_and_evaluate_code_block(code)

        if e is None:
            print('The performance of the pipeline is:', performance)
            valid_model = True
            pipeline_sentence = f"The code was executed correctly"
        else:
            valid_model = False
            pipeline_sentence = "The last code did not generate valid pipe, it was discarded."

        display_method(
            "\n"
            + f"*Error? : {str(e)}*\n"
            + f"*Valid model: {str(valid_model)}*\n"
            + f"```python\n{format_for_display(code)}\n```\n"
            + f"Performance {performance} \n"
            + f"{pipeline_sentence}\n"
            + f"\n"
        )
        if e is None:
            list_score_optimized.append(performance)
            list_pipelines_optimized.append(pipe)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(f'pipelines_{identifier}.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, code, str(performance)])

        iterations+=1

    index_max_performance = max(list_score_optimized)
    best_pipeline = list_pipelines_optimized[list_score_optimized.index(index_max_performance)]
    return best_pipeline

