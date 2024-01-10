from langchain.document_loaders import PyPDFLoader

# required `pip install pypdf`
loader = PyPDFLoader("csv_sample.pdf")
pages = loader.load_and_split()

print(pages[0].page_content)
"""
csv_sample
Page 1nameagecountry
Neville Hardy 56Niue
Dacia Cohen 74Falkland Islands (Malvinas)
Kathey Daniel 10Slovenia
Mallie Welch 12Equatorial Guinea
Katia Bryant 14Ghana
Laurice Robertson 53Saudi Arabia
Minh Barrett 27French Southern Territories
Latashia Perez 52Finland
Elvina Ross 68New Zealand
"""
