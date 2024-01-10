import pandas as pd
import matplotlib.pyplot as plt
import openai

# Replace with your OpenAI API key
openai.api_key = 

def generate_code(history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = history
    )

    return response['choices'][0]['message']['content']

def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

def create_plot(plot_code, data):
    exec(plot_code)

def main():
    file_path = "./whis_norm_inhouse_good.tsv"
    desired_columns = ["wer", "dataset", "provider"]
    data = read_tsv(file_path)
    data = data[desired_columns]
    history = [
        {"role": "system" , "content" : "You are a helpful assistant that specializes in making beautiful plots to understand tabular data in python. You only return working python code and assume that all data of interest will be contained in a pandas dataframe named data. You do not respond with any text that isn't executable code"},
        {"role": "user", "content" : f"You are a helpful assistant that specializes in making beautiful plots to understand tabular data in python. You can only return working python code and assume that all data of interest will be contained in a pandas dataframe named data. \nGenerate a python expression generates insightful plots from the dataframe contained in the variable data. some info on data is:\n{data.describe().to_string(index=False)}\n  data contains the columns : [wer, dataset, provider].\n Do not include any text that is not executable code in your response."}
    
    ]
    generated_code = generate_code(history)
    print("Generated code:\n", generated_code)

    while True:
        create_plot(generated_code, data)
        plt.show()

        prompt = input("Enter a natural language prompt to refine the plot or type 'exit' to quit: ").strip()
        if prompt.lower() == "exit":
            break
        else:
            history = history.append({"role": "assistant", "content" : generated_code})
            history = history.append({"role": "user", "content" : prompt})
        refined_code = generate_code(prompt)
        print("Refined code:\n", refined_code)
        generated_code = refined_code

if __name__ == "__main__":
    main()
