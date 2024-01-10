import pandas
import re, csv
from openai import OpenAI

df = pandas.read_csv("data/data_embeddings.csv", usecols=['pmid', 'abstract'])

df_already_processed_documents_pmid = pandas.read_csv("data/test_dataset.csv", usecols=["PMID"], sep='\t')
already_processed_documents_pmid = set(df_already_processed_documents_pmid['PMID'])

# print(already_processed_documents_pmid)

client = OpenAI(
    api_key = "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"
)


count = 0
seen_documents_pmid = set()
for i in range(df.shape[0]):
    if count == 1:
        break
    pmid, abstract = df.iloc[i, ]
    if pmid not in seen_documents_pmid and pmid not in already_processed_documents_pmid:
        seen_documents_pmid.add(pmid) # multiple columns contain the same abstract due to chunking
        prompt = "You generate questions such as multiple-choice questions, true/false questions, with the choices \
included in the question from the given abstract as well as their answers. For each question generate a python list of 3 elements \
like [], first element, type of question as string that is either 'T/F' or 'MC', the second element that is a string of question \
with the possible choices included, third is the  answer as a string that can be 'T', 'F', 'a', 'b','c', etc. \
The abstract is: " + abstract + "Remember and be careful: each of the entries in the lists should be a string with quotation marks!!"
        #  + "You just give a python list of lists of size 3 with the mentioned entities for each abstract at the end. That is like [[.., .., ..,], [.., .., ..,], ..]"
        # print(prompt)
        if prompt:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gpt-3.5-turbo-1106"
            )
        reply = chat_completion.choices[0].message.content
        print(reply)

        lists = re.findall(r'\[.*?\]', reply) # find the list

        if not lists: # did not reply in the correct form
            print("Warning: No lists found in the input string.")
            break
        # Convert strings to lists
        resulting_lists = [eval(lst) for lst in lists]

        # adding the questions to the testing set

        test_set_file_path = 'data/test_dataset.csv'

        with open(test_set_file_path, 'a', newline='') as file:
            csv_writer = csv.writer(file, delimiter='\t')
            for lst in resulting_lists:
                lst[1] = lst[1].replace('\n', ' ').replace('\t', ' ') # new lines and tabs in question choices arise issues

                new_record = [pmid, abstract] + lst 
                csv_writer.writerow(new_record)
        count += 1
