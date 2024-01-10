import openai

def get_artists(genre, label):
    openai.api_key = "KEY"
    print("label " + str(label))

    response = openai.Completion.create(
        engine="text-davinci-003",  # or the latest available engine
        prompt=f"Analyze this list of {genre} artists from {label} and rank them based on popularity: \n",
        max_tokens=100
    )

    # Split the response into lines and take the first five lines
    top_five_artists = response.choices[0].text.strip().split('\n')[:5]

    # Remove numbers and extra spaces from each artist's name
    artists = [artist.split('. ', 1)[-1].strip() for artist in top_five_artists]
    print(artists)
    return artists

# Example usage
# print(get_artists("Hip-Hop", "Some Label"))
