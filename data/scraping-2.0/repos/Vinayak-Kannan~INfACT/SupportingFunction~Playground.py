import time
import openai
import pandas as pd
from dotenv import dotenv_values
import numpy as np
import hdbscan

config = dotenv_values("/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/.env")
openai.api_key = config.get("SECRET_KEY")
columbia = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/KnowledgeOutputUpdated.csv')
embeddings = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/KnowledgeOutputv2.csv')
mit = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/MIT_Fall2023_v1/KnowledgeOutputUpdated.csv')
mit_embeddings = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/MIT_Fall2023_v1/KnowledgeOutputv2.csv')

industry = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Amazon/Amazon_KW_Embeddings.csv')
industry_outcome = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Amazon/Amazon knowledge Collapsed.csv')

# Drop all columns in embeddings except for 'Skill' and 'Embedding'
columbia_embeddings = embeddings[['Skill', 'Embedding']]
mit_embeddings = mit_embeddings[['Skill', 'Embedding']]
industry = industry[['Skill', 'Embedding']]

columbia_embeddings['Group'] = ''
mit_embeddings['Group'] = ''
industry['Group'] = ''



# Join embeddings to columbia on 'Skill' column
# columbia = columbia.join(embeddings.set_index('Skill'), on='Skill')

# Go through each row in df and convert 'Embedding' column to a list of embeddings
for i, row in columbia_embeddings.iterrows():
    embedding_text = row['Embedding']
    # Drop first and last character from embedding_text
    embedding_text = embedding_text[1:-1]
    # Split embedding_text by comma
    embedding_text = embedding_text.split(",")
    embedding_array = []
    for val in embedding_text:
        embedding_array.append(float(val))

    columbia_embeddings['Embedding'].update(pd.Series([embedding_array], index=[i]))

for i, row in mit_embeddings.iterrows():
    embedding_text = row['Embedding']
    # Drop first and last character from embedding_text
    embedding_text = embedding_text[1:-1]
    # Split embedding_text by comma
    embedding_text = embedding_text.split(",")
    embedding_array = []
    for val in embedding_text:
        embedding_array.append(float(val))

    mit_embeddings['Embedding'].update(pd.Series([embedding_array], index=[i]))

# for i, row in columbia.iterrows():
#     embedding_text = row['Embedding_Context']
#     # Drop first and last character from embedding_text
#     embedding_text = embedding_text[1:-1]
#     # Split embedding_text by comma
#     embedding_text = embedding_text.split(",")
#     embedding_array = []
#     for val in embedding_text:
#         embedding_array.append(float(val))
#
#     columbia['Embedding_Context'].update(pd.Series([embedding_array], index=[i]))

for i, row in industry.iterrows():
    embedding_text = row['Embedding']
    # Drop first and last character from embedding_text
    embedding_text = embedding_text[1:-1]
    # Split embedding_text by comma
    embedding_text = embedding_text.split(",")
    embedding_array = []
    for val in embedding_text:
        embedding_array.append(float(val))

    industry['Embedding'].update(pd.Series([embedding_array], index=[i]))


# Combine embeddings from columbia and industry
embeddings = columbia_embeddings['Embedding'].tolist() + industry['Embedding'].tolist() + mit_embeddings['Embedding'].tolist()
skills = columbia_embeddings['Skill'].tolist() + industry['Skill'].tolist() + mit_embeddings['Skill'].tolist()
pre_skills = pd.DataFrame(list(zip(skills, embeddings)), columns=['Skill', 'Embedding'])
# drop rows from pre_skills where 'Skill' is duplicate
pre_skills = pre_skills.drop_duplicates(subset=['Skill'])
embeddings = pre_skills['Embedding'].tolist()
skills = pre_skills['Skill'].tolist()


print(len(skills), len(embeddings))

hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, prediction_data=True).fit(embeddings)
print(len(embeddings))
print(hdb.labels_.max())
print(np.count_nonzero(hdb.labels_ == -1))
print(np.bincount(hdb.labels_[hdb.labels_ != -1]))
labels = hdb.labels_
membership_vectors = [np.argmax(x) for x in hdbscan.all_points_membership_vectors(hdb)]
max_vectors = np.asarray([max(x) for x in hdbscan.all_points_membership_vectors(hdb)])
# Delete elements from max_vector is index of element in hdb.labels_ does not equal -1
max_vectors = np.asarray(max_vectors[hdb.labels_ == -1])
avg_threshold = np.percentile(max_vectors, 25)
print(avg_threshold)
max_vectors = np.asarray([max(x) for x in hdbscan.all_points_membership_vectors(hdb)])



# for i in range(len(hdb.labels_)):
#     if i < len(columbia):
#         columbia_embeddings['Group'][i] = hdb.labels_[i]
#     elif i < len(columbia) + len(industry):
#         industry['Group'][i - len(columbia)] = hdb.labels_[i]
#     else:
#         mit_embeddings['Group'][i - len(columbia) - len(industry)] = hdb.labels_[i]

groups = hdb.labels_

# Create dataframe with columns Skills and Group using skills and groups lists
collapsed_skills = pd.DataFrame(list(zip(skills, groups)), columns=['Skill', 'Group'])
collapsed_skills['Collapsed Skill'] = ''

for i, row in collapsed_skills.iterrows():
    if row['Group'] == -1 and max_vectors[i] > avg_threshold:
        collapsed_skills.loc[i, 'Group'] = membership_vectors[i]

# Print count of rows where Group is -1
print("NEW: " + str(len(collapsed_skills[collapsed_skills['Group'] == -1])))

prompt = """
Summarize the following skills into a label that represent all the skills. The label can consist of multiple words. You should only output the new label with no other information or explanation

"""

count = 0
for i, row in collapsed_skills.iterrows():
    print(i)
    # Print count of rows where 'Collapsed Skill' column is not empty
    print(len(collapsed_skills[collapsed_skills['Collapsed Skill'] != '']))
    if collapsed_skills.loc[i, 'Group'] == -1:
        collapsed_skills.loc[i, 'Collapsed Skill'] = row['Skill']
    elif collapsed_skills.loc[i, 'Collapsed Skill'] != '':
        continue
    else:
        count += 1
        print(str(count) + " vs " + str(hdb.labels_.max()))
        # Get all instances of row['Skill'] where row['Group'] is equal to row['Group']
        instances = collapsed_skills[collapsed_skills['Group'] == row['Group']]['Skill'].tolist()
        print(instances)
        # Join instances into a string separated by commas
        instances = ', '.join(instances)
        response = [{
            "role": "user",
            "content": prompt + instances
        }]
        # Try to use ChatCompletion API. If no response after 30 seconds, try again
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=response,
                    temperature=0,
                    timeout=30
                )
                if response and response.choices and response.choices[0] and response.choices[0].message and response.choices[0].message.content:
                    break
                else:
                    print("Trying again")
                    time.sleep(1)
            except:
                print("Trying again")
                time.sleep(1)

        print(response.choices[0].message.content)
        # In all instances of row['Skill'] where row['Group'] is equal to row['Group'], set 'Collapsed Skill' to response.choices[0].message.content
        collapsed_skills.loc[collapsed_skills['Group'] == row['Group'], 'Collapsed Skill'] = response.choices[0].message.content

print(collapsed_skills)

# Join collapsed_skills 'Collapsed Skill' to columbia on 'Skill' column
# Drop 'Collapsed Skill' column from columbia
columbia = columbia.drop('Collapsed Skill', axis=1)
collapsed_skills = collapsed_skills.drop('Group', axis=1)
columbia = columbia.join(collapsed_skills.set_index('Skill'), on='Skill')
# Join collapsed_skills 'Collapsed Skill' to mit on 'Skill' column
mit = mit.drop('Collapsed Skill', axis=1)
mit = mit.join(collapsed_skills.set_index('Skill'), on='Skill')

industry_outcome = industry_outcome.drop('Collapsed Skill', axis=1)
industry_outcome = industry_outcome.join(collapsed_skills.set_index('Skill'), on='Skill')

columbia.to_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/KnowledgeOutputUpdatedAmazon11012023.csv', index=False)
mit.to_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/MIT_Fall2023_v1/KnowledgeOutputUpdatedAmazon11012023.csv', index=False)
# industry_outcome.to_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Amazon/Collapsed Skills11012023.csv', index=False)
industry_outcome.to_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Amazon/Collapsed Knowledge11012023.csv', index=False)



# for i, row in industry.iterrows():
#     embedding_text = row['Embedding_Context']
#     # Drop first and last character from embedding_text
#     embedding_text = embedding_text[1:-1]
#     # Split embedding_text by comma
#     embedding_text = embedding_text.split(",")
#     embedding_array = []
#     for val in embedding_text:
#         embedding_array.append(float(val))
#
#     industry['Embedding_Context'].update(pd.Series([embedding_array], index=[i]))


# output = hdb.labels_.tolist()
# # Compute the histogram of the array
# hist, bin_edges = np.histogram(output)
#
# # Plot the histogram
# plt.hist(output, bins=bin_edges, edgecolor='black')
# plt.xlabel('Integer')
# plt.ylabel('Count')
# plt.title('Histogram of Integers in Array')
# plt.show()


# columbia['Similar Skill'] = ''
# industry['Collapsed Skill'] = ''
# for i, r in industry.iterrows():
#     for j, r2 in columbia.iterrows():
#         if cosine_similarity(r['Embedding'], r2['Embedding']) > 0.83:
#             industry.loc[i, 'Collapsed Skill'] = r2['Collapsed Skill']
#
# # Drop all columns from industry except for 'Collapsed Skill' and 'Skill'
# industry = industry[['Collapsed Skill', 'Skill']]
# for i, row in industry_outcome.iterrows():
#     if row['Skill'] == row['Collapsed Skill'] and row['Skill'] in industry['Skill'].values:
#         # Find index of row in industry where 'Skill' column is equal to row['Skill']
#         index = industry.index[industry['Skill'] == row['Skill']].tolist()[0]
#         # Update 'Collapsed Skill' column in industry with value from industry_outcome['Collapsed Skill']
#         industry_outcome.loc[i, 'Collapsed Skill'] = industry['Collapsed Skill'][index]



# Save industry to csv
# industry_outcome.to_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Amazon/Collapsed Skills v2.csv', index=False)

# for i, r in columbia.iterrows():
#     for j, r2 in mit.iterrows():
#         if cosine_similarity(r['Embedding_Context'], r2['Embedding_Context']) > 0.85:
#             columbia.loc[i, 'Similar Skill Context'] = r2['Skill']
#             break
#
# mit['Similar Skill'] = ''
# mit['Similar Skill Context'] = ''
# for i, r in mit.iterrows():
#     for j, r2 in columbia.iterrows():
#         if cosine_similarity(r['Embedding'], r2['Embedding']) > 0.85:
#             mit.loc[i, 'Similar Skill'] = r2['Skill']
#             break
#
# for i, r in mit.iterrows():
#     for j, r2 in columbia.iterrows():
#         if cosine_similarity(r['Embedding_Context'], r2['Embedding_Context']) > 0.85:
#             mit.loc[i, 'Similar Skill Context'] = r2['Skill']
#             break

# Count number of values in 'Similar Skill' column that are not empty
# print("Columbia")
# print(len(columbia[columbia['Similar Skill'] != '']))
# print(len(columbia[columbia['Similar Skill Context'] != '']))
# print(len(columbia))
#
# print("MIT")
# print(len(mit[mit['Similar Skill'] != '']))
# print(len(mit[mit['Similar Skill Context'] != '']))
# print(len(mit))
#
# # Get all values in 'Skill' column that contain the word 'programming'
# print(columbia[columbia['Skill'].str.contains('Deep learning')]['Skill'])
# print(columbia[columbia['Skill'].str.contains('Deep learning')]['Similar Skill'])

# Print number of values in 'Collapsed Skill' column that are not empty

# # Load in data from csv industry collapsed skills
# industry = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Industry/Collapsed Skills.csv')
# # Load in data from columbia skill output
# columbia = pd.read_csv('/Users/vinayakkannan/Desktop/INfACT/Script/SupportingFunction/RawData/Columbia_Fall2023_v1/SkillOutputUpdated.csv')
# # Print Skill and 'Collapsed Skill' columns from industry where 'Collapsed Skill' value is in columbia 'Collapsed Skill' column
# print(industry[industry['Collapsed Skill'].isin(columbia['Collapsed Skill'])]['Collapsed Skill'])