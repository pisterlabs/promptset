import csv
import openai
import time

query_list = []
label_list = []
prompt_template = "Please determine whether this string contains pornographic content as follows: '{}'. Answer 'yes' if this string contains pornographic content, otherwise answer 'no'. Please answer:"


if __name__ == "__main__":

    with open('frameIo_dataset.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)

        for row in csvreader:
            query, label = row[0], row[1]
            query = query[12:]

            if row[0] != "" and row[1] != "":
                query_list.append(query)
                label_list.append(label)  

    print(len(query_list), len(label_list))
    count = 0

    with open('frameIo_dataset_cleansed.csv', 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['Query', 'Label'])

        for query, label in zip(query_list, label_list):

            prompt = prompt_template.format(query)

            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                temperature=0.5,
            )
            output = response.choices[0].text.strip().lower()

            if output == "no":   
                csv_writer.writerow([query, label])
                count += 1
            elif output == "yes":
                print('pornographic query detected:', query)
            else:
                print("None.")
    
    print(count)