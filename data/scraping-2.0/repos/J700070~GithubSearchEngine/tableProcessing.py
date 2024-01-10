import os
import openai




def processTable(table, query,ai_activated, required_words, words_to_avoid, min_star_limit, show_private, sort, ascending):
    """Process the table and return only relevant entries"""
    if ai_activated:
        openai.api_key = os.getenv("OPENAI_API_KEY")

    print(f"Initial size: {len(table)}")

    # Filter the results to find the relevant repositories based on custom criteria
    table = table.sort_values(by=sort, ascending=ascending)

    # Remove duplicates
    table = table.drop_duplicates(subset=["Name", "Repository Name", "Created Date", "Language", "Stars", "Forks Count", "Score", "Private", "Owner Name", "URL", "Description", "Size"], keep="first")
    table = table.reset_index(drop=True)

    print(f"Size after removing duplicates: {len(table)}")

    # Remove repositories with less starts that the min_star_limit
    table = table[table["Stars"] >= min_star_limit]

    print(f"Size after removing repositories with less starts that the min_star_limit: {len(table)}")

    # Remove private repositories
    if not show_private:
        table = table[table["Private"] == False]
    
    print(f"Size after removing private repositories: {len(table)}")

    # Remove repositories that do not contain all of the required_words or have no description
    if len(required_words) > 0:
        table = table[table["Description"].notna()]

    for word in required_words:
        # Search for the word in the description, the name or the topics
        table = table[table["Description"].str.contains(word, case=False) | table["Name"].str.contains(word, case=False) | table["Topics"].str.contains(word, case=False)]

    print(f"Size after removing repositories that do not contain all of the required_words or have no description: {len(table)}")

    # Remove repositories that contain words to avoid
    for word in words_to_avoid:
        # Search for the word in the description, the name, the topics or the language.
        table = table[~table["Description"].str.contains(word, case=False, na=False)]
        table = table[~table["Name"].str.contains(word, case=False, na=False)]
        table = table[~table["Topics"].str.contains(word, case=False, na=False)]
        table = table[~table["Language"].str.contains(word, case=False, na=False)]

    print(f"Size after removing repositories that contain words to avoid: {len(table)}")
    
    if ai_activated:
        # Make a new column with the score of based on openai input
        table["AI_Score"] = 0

        for index, row in table.iterrows():
            # Make markdown text for the title and description on a single string
            title = f"# {row['Name']}"
            description = f"## {row['Description']}"
            
            query = f"## Objective: \n {query}"

            # Create the openai input
            openai_input = f"{title}\n{description}\n{query}\n Score the repo relevance for the query from 1 to 10: \n"

            # response = openai.Completion.create(
            #     model="text-babbage-001",
            #     prompt=openai_input,
            #     max_tokens=1,
            #     temperature=0,
            # )

            # txt_response = response["choices"][0]["text"]
            txt_response = "1"

            # Make sure the response is valid
            if not txt_response.isdigit() or len(txt_response.strip()) == 0:
                # print("Error: OpenAI did not return a valid response.")
                table.at[index, "AI_Score"] = -1
            else:
                # Save the score in the table
                table.at[index, "AI_Score"] = int(txt_response)

        # Sort the table by score
        table = table.sort_values(by="AI_Score", ascending=False)

        

    return table

