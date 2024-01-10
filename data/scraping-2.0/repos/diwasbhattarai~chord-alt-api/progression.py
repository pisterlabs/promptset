import openai

response = openai.ChatCompletion.create(
  model="gpt-4",
  messages=[
        {"role": "system", "content": 
         '''You are a helpful assistant who is an expert on building computer systems in the context of music applications. The application you will help build would ideally ask users their current chord progressions and based on it you will answer how can the same progression be reharmonized keeping the vibe same. The underlying engine would be a LLM that would take user's progression and ouputs reharmonized version of the original progression. 

Make sure to list five different chord progressions in the [style] given by the user in mind. The list should display progressions that are sorted by difficulty. All progression should be based on the original progression and match the vibe as much as possible.

Make sure to answer in consistent JSON format to use the output in an application. Do not include introduction and conclusion part in your answer. Only strictly respond in valid JSON format. The format should be:

{
'original_progression': [progression],
'alternates': [
'reharmonized': [reharmonized progression], 'explanation':[explanation], 'new_passing_chords':[new passing chords], 'passing_chord_locations':[passing chord location1, passing chord location2..], 'passing_chord_explanation':[passing chord explanation],
...
]
}

The prompt looks like the following:
Hey GPT-4, can you give me a list of new chords that could be used to reharmonize the chord progression [progression] in [style]? Make sure you list five unique chord progressions and also provide a long and detailed explanation with music theory of what the new progression means and how it relates to the original progression. Also include new passing chords depending on 'include passing chords' field in user's query and list them in the response json field 'passing' whenever they are present in the reharmonized progression. Also include the location of passing chords as a tuple of chords from the original chords between which we should insert this new passing chord. Additionally, provide explanation as to why those passing chord makes sense from the music theory point of view.'''},

        {"role": "user", "content": "progression = [G, Em, C, D] style = [jazz, funk, rnb] include passing chords = yes"},
        
    ]
)

print(response.choices[0]['message']['content'])
print(response)