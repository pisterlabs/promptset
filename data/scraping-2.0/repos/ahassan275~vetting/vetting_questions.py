from langchain.document_loaders.csv_loader import CSVLoader
import re


loader = CSVLoader(file_path="_Software Vetting Request and Resolution List - SW Vetting Template.csv")
data = loader.load()

extracted_questions = [row.page_content.split('\n: ')[1].split('\n')[0] for row in data[31:74]]

# extracted_fields = []
# for string in extracted_questions:
#     match = re.search(r'(\d+\.\d+)\s+(.*)', string)
#     if match:
#         numbered_field = match.group(1)
#         text = match.group(2)
#         extracted_fields.append((numbered_field, text))
#
# extracted_dict_list = [{'number': field, 'question': text} for field, text in extracted_fields]
print(extracted_questions)