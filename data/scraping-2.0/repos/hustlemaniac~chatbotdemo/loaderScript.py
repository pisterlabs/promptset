from libraries import *
# from langchain.document_loaders import *
def loaderCall():
    # print(globals())
    listDocLoaders = [i for i in globals() if 'Loader' in i ]
    print(f'{len(listDocLoaders)} loaders have been retrieved')
    print("**** Loaders available ****")
    for i in range(len(listDocLoaders)):
        print(f"{i + 1}. {listDocLoaders[i]}")
    loaderChoosen = input("Choose a loader(mind the spelling) : ")
    while loaderChoosen not in listDocLoaders:
        if loaderChoosen.lower() == "quit":
            sys.exit("User chose to quit")
        print("**** Loader not found ****")
        loaderChoosen = input("Choose a loader(mind the spelling), type quit to exit: ")
    print(f"You have choosen {loaderChoosen}")
    in_source = input("Give appropriate source : ")
    loader_class = globals()[loaderChoosen]
    loader = loader_class(in_source)
    return loader
