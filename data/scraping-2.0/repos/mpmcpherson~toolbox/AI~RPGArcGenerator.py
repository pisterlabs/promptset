import openai

# Set your OpenAI GPT-3.5 API key here
openai.api_key = "YOUR_API_KEY"

def generate_summary_and_next_arc(previous_arcs):
    # Generate a summary of previous plot arcs
    summary_prompt = "Summarize the previous RPG plot arcs:\n\n".join(previous_arcs)
    summary_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=summary_prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    summary = summary_response.choices[0].text.strip()

    # Use the summary as a prompt to generate the next plot arc
    next_arc_prompt = f"Generate an RPG plot arc based on the following summary:\n{summary}"
    next_arc_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=next_arc_prompt,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.8,
    )
    next_arc = next_arc_response.choices[0].text.strip()

    return summary, next_arc

if __name__ == "__main__":
    # Load previous plot arcs from a file
    with open("previous_plot_arcs.txt", "r") as file:
        previous_arcs = file.readlines()

    # Generate summary and the next plot arc
    summary, next_arc = generate_summary_and_next_arc(previous_arcs)

    # Print the results
    print("\nSummary of Previous Plot Arcs:")
    print(summary)
    print("\nGenerated Next RPG Plot Arc:")
    print(next_arc)

    # Save the new plot arc to the file for future use
    with open("previous_plot_arcs.txt", "a") as file:
        file.write(next_arc + "\n")
