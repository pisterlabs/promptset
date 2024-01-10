from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='./csv_sample.csv')
data = loader.load()
print(type(data[0]))
"""
<class 'langchain.schema.document.Document'>
"""

print(data)
"""
[Document(page_content='name: Neville Hardy\nage: 56\ncountry: Niue', metadata={'source': './csv_sample.csv', 'row': 0}), Document(page_content='name: Dacia Cohen\nage: 74\ncountry: Falkland Islands (Malvinas)', metadata={'source': './csv_sample.csv', 'row': 1}), Document(page_content='name: Kathey Daniel\nage: 10\ncountry: Slovenia', metadata={'source': './csv_sample.csv', 'row': 2}), Document(page_content='name: Mallie Welch\nage: 12\ncountry: Equatorial Guinea', metadata={'source': './csv_sample.csv', 'row': 3}), Document(page_content='name: Katia Bryant\nage: 14\ncountry: Ghana', metadata={'source': './csv_sample.csv', 'row': 4}), Document(page_content='name: Laurice Robertson\nage: 53\ncountry: Saudi Arabia', metadata={'source': './csv_sample.csv', 'row': 5}), Document(page_content='name: Minh Barrett\nage: 27\ncountry: French Southern Territories', metadata={'source': './csv_sample.csv', 'row': 6}), Document(page_content='name: Latashia Perez\nage: 52\ncountry: Finland', metadata={'source': './csv_sample.csv', 'row': 7}), Document(page_content='name: Elvina Ross\nage: 68\ncountry: New Zealand', metadata={'source': './csv_sample.csv', 'row': 8})]
"""

print(data[0].page_content)
"""
name: Neville Hardy
age: 56
country: Niue
"""

print(data[0].metadata)
"""
{'source': './csv_sample.csv', 'row': 0}
"""

loader = CSVLoader(
    file_path='./csv_sample_no_header.csv',
    csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['name', 'age', 'country']
    })

data = loader.load()
print(data[1].page_content)
"""
name: Dacia Cohen
age: 74
country: Falkland Islands (Malvinas)
"""