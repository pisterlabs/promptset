import openai
import json

def get_completion(prompt):
    inference = openai.Completion.create(
    #model="curie:ft-llm-hackathon-synthesis-parsing-team-2023-03-29-22-57-05",
    model="davinci:ft-llm-hackathon-synthesis-parsing-team-2023-03-30-00-54-58",
    prompt=prompt,
    max_tokens=1000,
    stop=["###"])

    print(inference)
    
    completion = json.loads(inference["choices"][0]["text"])

    return completion

prompt = "To a solution of 2050 mg of the ketoxime (VI) in 25 ml of ethylene glycol, was added 0.7 ml of 80% hydrazine hydrate. After having been kept at 40\u00b0 to 50\u00b0 C. for 30 minutes, the solution was admixed with 750 mg of potassium hydroxide and kept at 150\u00b0 C. for 4 hours under an argon atmosphere. After cooling and adding 50 ml of water, the solution was neutralized with 1 N HCl and extracted with methylene chloride. The extract was washed with an aqueous sodium bicarbonate solution, then with an aqueous sodium chloride solution, dried over magnesium sulfate, and concentrated in vacuo to give 970 mg of a crystalline compound (V). M.p. 130\u00b0-133\u00b0 C.; MS m/e: 183 (M+).\n\n###\n\n"

print(get_completion(prompt))