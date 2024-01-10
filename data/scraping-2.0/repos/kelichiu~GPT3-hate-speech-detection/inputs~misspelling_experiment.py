import openai
import pandas as pd
import time
import inputs.data_collection_functions as dc
openai.organization = "INSERT ORG ID"
openai.api_key = "INSERT API KEY"

ethos_data = pd.read_csv("inputs/Ethos_Dataset_Multi_Label.csv", delimiter=';')
ethos_data_binary = pd.read_csv("inputs/Ethos_Dataset_Binary.csv", delimiter=';')
sexist = pd.DataFrame(ethos_data[ethos_data["gender"] >= 0.5]["comment"])       # Filter ethos_data for gender score > 0.5, take only comment column (84 rows)
racist = pd.DataFrame(ethos_data[ethos_data["race"] >= 0.5]["comment"])         # Filter ethos_data for race score > 0.5, take only comment column (76 rows)
not_hate = ethos_data_binary[ethos_data_binary["isHate"] == 0]                  # Filter ethos_data_binary for non-hate comments (354 rows)
sexist["category"] = "sexist"       # Add "category" column to comments
racist["category"] = "racist"       # ""
not_hate["category"] = "not hate"   # ""
comments = pd.concat([sexist, racist])  # Create one dataframe called comments (160 rows)
comments = comments.reset_index()

profanity_sexist = pd.DataFrame()
profanity_racist = pd.DataFrame()

profanity_sexist = profanity_sexist.append([sexist[sexist["comment"].str.contains("nigg")],
                                             sexist[sexist["comment"].str.contains("fuck")],
                                             sexist[sexist["comment"].str.contains("phuck")],   # Specific example from paper
                                             sexist[sexist["comment"].str.contains("whore")],
                                             sexist[sexist["comment"].str.contains("kill")],
                                             sexist[sexist["comment"].str.contains("rape")],
                                             sexist[sexist["comment"].str.contains("die")],
                                             sexist[sexist["comment"].str.contains("bitch")]])

profanity_racist = profanity_racist.append([racist[racist["comment"].str.contains("nigg")],
                                             racist[racist["comment"].str.contains("fuck")],
                                             racist[racist["comment"].str.contains("phuck")],   # Specific example from paper
                                             racist[racist["comment"].str.contains("whore")],
                                             racist[racist["comment"].str.contains("kill")],
                                             racist[racist["comment"].str.contains("rape")],
                                             racist[racist["comment"].str.contains("die")],
                                             racist[racist["comment"].str.contains("bitch")]])

# Drop duplicates
profanity_sexist = profanity_sexist.drop_duplicates()
profanity_racist = profanity_racist.drop_duplicates()

# Create non-profanity data set by dropping all indexes in the original data set that appear in the profanity data set
non_profanity_sexist = sexist.drop(profanity_sexist.index)
non_profanity_racist = racist.drop(profanity_racist.index)
non_profanity_racist.to_csv("outputs/data/non_profanity_racist.csv")
non_profanity_sexist.to_csv("outputs/data/non_profanity_sexist.csv")

profanity_sexist["status"] = "unedited"
profanity_racist["status"] = "unedited"
profanity_sexist["comment_id"] = profanity_sexist.reset_index().index
profanity_racist["comment_id"] = profanity_racist.reset_index().index

# Define strings to search for (case insensitive) and misspellings/censorship to replace with
profane_words = [["(?i)nigg", "n1gg"],
                 ["(?i)nigg", "n*gg"],
                 ["(?i)nigg", "nig"],
                 ["(?i)nigg", "n-gg"],
                 ["(?i)phuck", "fuck"],
                 ["(?i)fuck", "fck"],
                 ["(?i)fuck", "phuck"],
                 ["(?i)fuck", "f*ck"],
                 ["(?i)fuck", "f-ck"],
                 ["(?i)whore", "wh0re"],
                 ["(?i)whore", "wh0r3"],
                 ["(?i)whore", "whor3"],
                 ["(?i)whore", "wh-re"],
                 ["(?i)whore", "wh*re"],
                 ["(?i)kill", "k1ll"],
                 ["(?i)kill", "kil"],
                 ["(?i)kill", "k-ll"],
                 ["(?i)kill", "k*ll"],
                 ["(?i)rape", "r@pe"],
                 ["(?i)rape", "rap"],
                 ["(?i)rape", "r@p3"],
                 ["(?i)rape", "rap3"],
                 ["(?i)rape", "r*pe"],
                 ["(?i)rape", "r-pe"],
                 ["(?i)bitch", "b1tch"],
                 ["(?i)bitch", "b-tch"],
                 ["(?i)bitch", "b*tch"],
                 ["(?i)bitch", "bich"],
                 ["(?i)bitch", "bicht"],
                 ["(?i)bitch", "bithc"],
                 ["(?i)die", "dye"],
                 ["(?i)die", "d1e"],
                 ["(?i)die", "di3"],
                 ["(?i)die", "d13"],
                 ["(?i)die", "d*e"],
                 ["(?i)die", "d-e"]]

profanity_sexist_edited = pd.DataFrame()
profanity_racist_edited = pd.DataFrame()

# 1. Loop through every word combination in profane words
# 2. Create a copy of the original data set with this misspelling introduced
# 3. Append it to the edited dataframe
for words in profane_words:
    sexist_edited_new = profanity_sexist.replace(to_replace=words[0], value=words[1], regex = True)
    racist_edited_new = profanity_racist.replace(to_replace=words[0], value=words[1], regex = True)
    profanity_sexist_edited = profanity_sexist_edited.append(sexist_edited_new)
    profanity_racist_edited = profanity_racist_edited.append(racist_edited_new)

# Add "edited" status
profanity_sexist_edited["status"] = "edited"
profanity_racist_edited["status"] = "edited"

# Drop duplicates
profanity_sexist_edited = profanity_sexist_edited.drop_duplicates()
profanity_racist_edited = profanity_racist_edited.drop_duplicates()

# Put edited and unedited results into one dataframe
profanity_sexist = profanity_sexist.append(profanity_sexist_edited)
profanity_racist = profanity_racist.append(profanity_racist_edited)
profanity_sexist.to_csv("outputs/data/profanity_sexist.csv")
profanity_racist.to_csv("outputs/data/profanity_racist.csv")


### Zero-shot ###
all_zero_shot_result = pd.DataFrame()   # Create empty data frame

# Sexist, temperature = 0.3
for i, sexist_comment in enumerate(profanity_sexist.comment):  # Loop through first 30 comments in sexist, where i is index number (0-29) and sexist_comment is comment as a string
    print("Sexist comment misspelling {} of {}".format(i, len(profanity_sexist)))
    zero_shot_result = dc.zero_shot("sexist", "sexist", sexist_comment, temperature = 0)    # Run sexist_comment through GPT-3
    zero_shot_result["status"] = profanity_sexist.reset_index().status[i]
    zero_shot_result["comment_id"] = profanity_sexist.reset_index().comment_id[i]
    all_zero_shot_result = all_zero_shot_result.append(zero_shot_result, ignore_index=True)  # Add results for sexist_comment to all results

# Racist, temperature = 0.3
for i, racist_comment in enumerate(profanity_racist.comment):
    print("Racist comment misspelling {} of {}".format(i, len(profanity_sexist)))
    zero_shot_result = dc.zero_shot("racist", "racist", racist_comment, temperature = 0)
    zero_shot_result["status"] = profanity_racist.reset_index().status[i]
    zero_shot_result["comment_id"] = profanity_racist.reset_index().comment_id[i]
    all_zero_shot_result = all_zero_shot_result.append(zero_shot_result, ignore_index=True)

### One-shot ###
all_one_shot_result = pd.DataFrame()
# Sexist, temperature = 0.3
for i, sexist_comment in enumerate(profanity_sexist.comment):
    # Extract comment_id of sexist_comment and use to index profanity_sexist and extract consistent example for misspelled versions of comments
    example = non_profanity_sexist.reset_index().comment[profanity_sexist.loc[profanity_sexist["comment"] == sexist_comment, "comment_id"].iloc[0]]
    one_shot_sexist_result = dc.one_shot("sexist", "sexist", example, sexist_comment, temperature=0.3)
    one_shot_sexist_result["status"] = profanity_sexist.reset_index().status[i]
    one_shot_sexist_result["comment_id"] = profanity_sexist.reset_index().comment_id[i]
    all_one_shot_result = all_one_shot_result.append(one_shot_sexist_result, ignore_index=True)

# Racist, temperature = 0.3
for i, racist_comment in enumerate(profanity_racist.comment):
    example = non_profanity_racist.reset_index().comment[profanity_racist.loc[profanity_racist["comment"] == racist_comment, "comment_id"].iloc[0]] # Use modulo so examples from the non-profanity list are recycled
    one_shot_racist_result = dc.one_shot("racist", "racist", example, racist_comment, temperature=0.3)
    one_shot_racist_result["status"] = profanity_racist.reset_index().status[i]
    one_shot_racist_result["comment_id"] = profanity_racist.reset_index().comment_id[i]
    all_one_shot_result = all_one_shot_result.append(one_shot_racist_result, ignore_index=True)


### Few shot, fixed examples ###
all_few_shot_fixed_examples_result = pd.DataFrame()

#Mixed Cat, temperature = 0.3
for i in range(0, 10):
    for j, racist_comment in enumerate(profanity_racist.comment):
        few_shot_fixed_examples_result = dc.few_shot_fixed_examples(i, "fixed-example", "racist", racist_comment, temperature = 0.3)
        few_shot_fixed_examples_result["status"] = profanity_racist.reset_index().status[j]
        few_shot_fixed_examples_result["comment_id"] = profanity_racist.reset_index().comment_id[j]
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(.5)
    for j, sexist_comment in enumerate(profanity_sexist.comment):
        few_shot_fixed_examples_result = dc.few_shot_fixed_examples(i, "fixed-example", "sexist", sexist_comment, temperature = 0.3)
        few_shot_fixed_examples_result["status"] = profanity_sexist.reset_index().status[j]
        few_shot_fixed_examples_result["comment_id"] = profanity_sexist.reset_index().comment_id[j]
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(.5)
    for j, sexist_comment_unedited in enumerate(non_profanity_sexist.comment):
        few_shot_fixed_examples_result = dc.few_shot_fixed_examples(i, "fixed-example", "sexist", sexist_comment_unedited, temperature = 0.3)
        few_shot_fixed_examples_result["status"] = non_profanity_sexist.reset_index().status[j]
        few_shot_fixed_examples_result["comment_id"] = non_profanity_sexist.reset_index().comment_id[j]
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(.5)
    for j, racist_comment_unedited in enumerate(non_profanity_racist.comment):
        few_shot_fixed_examples_result = dc.few_shot_fixed_examples(i, "fixed-example", "racist", racist_comment_unedited, temperature = 0.3)
        few_shot_fixed_examples_result["status"] = non_profanity_racist.reset_index().status[j]
        few_shot_fixed_examples_result["comment_id"] = non_profanity_racist.reset_index().comment_id[j]
        all_few_shot_fixed_examples_result = all_few_shot_fixed_examples_result.append(few_shot_fixed_examples_result, ignore_index=True)
        time.sleep(.5)


### Save results ###
all_zero_shot_result.to_csv("outputs/data/zero_shot_misspelling.csv")
all_one_shot_result.to_csv("outputs/data/one_shot_misspelling.csv")
all_few_shot_fixed_examples_result.to_csv("../outputs/data/few_shot_fixed_examples_misspelling.csv")
