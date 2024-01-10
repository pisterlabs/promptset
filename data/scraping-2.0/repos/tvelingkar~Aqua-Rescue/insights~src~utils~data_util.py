from langchain.document_loaders.csv_loader import CSVLoader

def loadData(dataDirPath, fileName):
    loader = CSVLoader(file_path=dataDirPath + fileName + '.csv')
    return loader.load()
