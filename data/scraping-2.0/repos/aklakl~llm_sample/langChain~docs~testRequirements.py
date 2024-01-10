import sys
print(sys.executable)

try:
    import langchain
    print("langchain is installed.")
except ImportError:
    print("langchain is not installed.")
    
    
try:
    from langchain.document_loaders import UnstructuredURLLoader
    print("langchain.document_loaders | UnstructuredURLLoader is installed.")
except ImportError:
    print("langchain.document_loaders | UnstructuredURLLoader is not installed.")

    
try:
    from langchain.document_loaders import SeleniumURLLoader
    print("SeleniumURLLoader is installed.")
except ImportError:
    print("SeleniumURLLoader is not installed.")

try:
    import unstructured
    print("unstructured is installed.")
except ImportError:
    print("unstructured is not installed.")
    

 #source path/to/your/virtualenv/bin/activate
 #/Users/ming/anaconda3/bin/python
