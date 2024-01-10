import requests
from bs4 import BeautifulSoup

TEST=False
SEP="-+-+-+-+"
#
WebFolder="general"
#
EmbeddingFileName="data/"+WebFolder+"_embed.npz" # Updated in embedweb() 
TextListFile="data/"+WebFolder+"_textlist.txt" # Updated in embedweb() 
DefaultLLM="text-embedding-ada-002"
AltLLM="gpt-3.5-turbo-1106"

def embeddingfilename(webfolder):
    return "data/"+webfolder+"_embed.npz"

# The following function is used to get the content of a web page
# The `url` is the URL address and the `filename` is the name of the file to save
# Remove all HTML tags and save the content to the file
def rdweb(url, filename):
    # Get the content of the web page
    r = requests.get(url)
    # Parse the content of the web page
    soup = BeautifulSoup(r.content, "html.parser")
    # Get the text of the web page
    text = soup.get_text()
    # Save the text to the file
    save=(not filename==None)
    if save:
        with open(filename, "w") as f:
            f.write(text)
    return text

if TEST:
    (url1, filename1)="https://halimgur.substack.com/p/the-requiem-for-a-dream-israels-untaken", "data/requiem.txt"
    sep="-+-+-+-+"
    text=rdweb(url1, filename1, save=False)

# The following function takes a text string `text` and separates it to a list
# of strings using the separator `sep`
def sepstr(text, sep):
    # Split the text to a list of strings
    lst=text.split(sep)
    # Remove the empty strings
    lst=[s for s in lst if s]
    # Remove the first string
    lst=lst[1:]
    return lst

if TEST:
    sa=sepstr(text, SEP)
    for (i, s) in enumerate(sa):
        print("Section %d\n" % (i+1))
        print(s)
        print()

# Establish OpenAI API key (see below for how to get one)
import os
import openai
from openai import OpenAI
import numpy as np
Client=None
LLM=None
OpenAI_KEY_notset="\n\nERROR *** OPENAI_API_KEY is not set. If you have an OpenAI key, let this \
program know about it:\n\1. Create the file .env in the project folder\n\
2. Enter the line OPENAI_API_KEY='my-api-key-here' in the .env file\n\
3. Add `.env` to the .gitignore file so that the key is not shared with others"
      
def set_openai_key():
    global Client, LLM
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key==None:
        print(OpenAI_KEY_notset)
        return
    Client = OpenAI()
    LLM=DefaultLLM

def init_openai(model=DefaultLLM):
    global Client, LLM
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key==None:
        print(OpenAI_KEY_notset)
        return
    Client = OpenAI()
    LLM=model
    return (Client, LLM)

# Generate the embedding for the text
def get_embedding(text):
    global Client, LLM
    if Client==None:
        set_openai_key()
    response = Client.embeddings.create(
        input=text,
        model=LLM
    )
    return np.array(response.data[0].embedding)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def read_page_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    page_list=[]
    for i in range(0,len(lines),2):
        page_list.append((lines[i].strip(),lines[i+1].strip()))
    return page_list

def embedweb(webfolder=WebFolder, model=DefaultLLM):
# This function embeds all the texts in the web folder
    # It first creates the string TextListFile that contains the list of URLs and the list of texts
    # It then creates the string EmbeddingFileName that will be the file to take the embeddings and the metadata
    # It reads the list of URLs and the list of texts from the file TextListFile as pairs of strings
    # The first string in each pair is the URL and the second string is the text
    # It then reads the text from the URL and separates it to a list of strings using the separator SEP
    # It then embeds each string in the list and appends the embedding to the list of embeddings
    # It then appends the index of the text and the index of the string in the text to the embedding
    # It then saves the embeddings to the file EmbeddingFileName
    # It uses the metadata dictionary to store the model, the web folder, the text list, and the texts
    # It then returns the embeddings and the metadata
    global Client, LLM, EmbeddingFileName, TextListFile
    if Client==None or model!=LLM:
        init_openai(model=model)
    TextListFile="data/"+webfolder+"_textlist.txt"
    EmbeddingFileName="data/"+webfolder+"_embed.npz"
    page_list=read_page_list(TextListFile)
    embed_list=[]
    for (itext,pair) in enumerate(page_list):
        url=pair[0]
        text=rdweb(url, None)
        sa=sepstr(text, SEP)
        for (isegment, s) in enumerate(sa):
            v=get_embedding(s)
            w=np.append(v,[itext,isegment])
            embed_list.append(w)
    embeddings=np.array(embed_list)
    # Save the embeddings to a file
    metadata={"model":LLM, "webfolder":webfolder, "textlist":TextListFile, "texts":page_list}
    np.savez(EmbeddingFileName,data=embeddings, metadata=metadata)

def load_embeddings(webfolder=WebFolder):
# This function loads the embeddings from the file
# The embeddings are returned as a numpy array
# The metadata is returned as a dictionary
# We need the `allow_pickle=True` option to load the metadata
# Otherwise, the function will refuse to load the metadata due to security reasons
    global EmbeddingFileName
    EmbeddingFileName="data/"+webfolder+"_embed.npz"
    with np.load(EmbeddingFileName, allow_pickle=True) as data:
        embeddings=data['data']
        metadata=data['metadata'].item()
    return (embeddings, metadata)

def update_embeddings(webfolder=WebFolder, model=DefaultLLM):
    # This function updates the embeddings for the web folder.
    # It checks if there is an embedding file for the web folder specified.
    # If not, it creates the embedding file.
    # If there is an embedding file, it checks the follolwing:
    # 1. If the model is not the same, it creates embeddings and returns
    # 2. It checks the list of text titles in the metadata against the text titles in TextListFile
    # 3. If the lists are the same, it returns
    # 4. For the new titles in the TextListFile,
    #.   it creates embeddings, updates the metadata, saves the embeddings, and returns
    global EmbeddingFileName, TextListFile
    EmbeddingFileName="data/"+webfolder+"_embed.npz"
    TextListFile="data/"+webfolder+"_textlist.txt"
    if not os.path.exists(EmbeddingFileName):
        embedweb(webfolder=webfolder, model=model)
        return True
    (embeddings, metadata)=load_embeddings(webfolder=webfolder)
    if not metadata["model"]==model:
        embedweb(webfolder=webfolder, model=model)
        return True
    page_list=read_page_list(TextListFile)
    embedded_texts=[]
    for pair in metadata["texts"]:
        embedded_texts.append(pair[1])
    update=False
    for (itext,pair) in enumerate(page_list):
        url=pair[0]
        texttitle=pair[1]
        # print("Check '"+texttitle+"' --> ", end="")
        if texttitle not in embedded_texts:
            print("Update "+texttitle)
            text=rdweb(url, None)
            sa=sepstr(text, SEP)
            for (isegment, s) in enumerate(sa):
                v=get_embedding(s)
                w=np.append(v,[itext,isegment])
                embeddings=np.vstack((embeddings,w))
            metadata["texts"].append(pair)
            update=True
        # else:
            # print("OK")
    if update:
        # Save the embeddings to a file
        np.savez(EmbeddingFileName,data=embeddings, metadata=metadata)
    return update

def showmetadata(metadata):
# This function shows the metadata of the embeddings
# The metadata is a dictionary with the following keys:
# "model", "webfolder", "textlist", "texts"
    print("Model:", metadata["model"])
    print("Web Folder:", metadata["webfolder"])
    print("Text List:", metadata["textlist"])
    print("Texts:")
    for (i, pair) in enumerate(metadata["texts"]):
        print("%d: %s" % (i+1, pair[1]))

# The following function sorts the np array `vsim` from highest to lowest and returns the sorted indices
def sort_indices(a):
    return np.argsort(a)[::-1]

def getsimilar(embeddings, query):
# This function gets the embedding of the question and finds the most similar embedding
    query_embedding=get_embedding(query)
    vsim=np.zeros(len(embeddings))
    max_similarity=0
    for i in range(len(embeddings)):
        w=embeddings[i]
        v=w[:-2]
        similarity = cosine_similarity(v, query_embedding)
        vsim[i]=similarity
    sorted_indices=sort_indices(vsim)
    return sorted_indices    

# The following function opens the text file `filename`. This file is orgaanised as pairs
# of lines. The first line in each pair is the url address and the second line is the
# title of the page.  The function reads the `n`th pair of lines and returns the url and
# the title.
def read_page(filename,n):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines[2*n].strip(),lines[2*n+1].strip()


# The following function takes an index array and returns the concatenetion of the top N strings
def gettopN(embeddings, metadata, indices, N=3):
    textlist=metadata["texts"]
    concat_text=""
    for x in indices[0:N]:
        jpage=int(embeddings[x][-2])
        jsegment=int(embeddings[x][-1])
        (url, title)=read_page(TextListFile,jpage)
        pagetext=rdweb(url, None)
        paginatedtext=sepstr(pagetext, SEP)
        text=paginatedtext[jsegment]
        concat_text=concat_text+text+"\n\n"
    return concat_text

def main():
    # The following two lines load the local environment variables from the file .env
    # The .env file is not shared with others because it is listed in the .gitignore file
    # If you have an OpenAI key, let this program know about it:
    # 1. Create the file .env in the project folder
    # 2. Enter the line OPENAI_API_KEY='my-api-key-here' in the .env file
    import dotenv
    dotenv.load_dotenv()
    webfolder="short"
    # The following line checks if there is an embedding file for the web folder specified.
    # If not, it creates the embedding file.
    if not os.path.exists(embeddingfilename(webfolder)):
        print("\n\nEmbeddings not found")
        embedweb(webfolder=webfolder, model=DefaultLLM)
        print("Embeddings created")
    #
    # The following updates the embeddings if there are new texts in the text list:
    print("\n\nCheck if we need to update the embeddings")
    update=update_embeddings(webfolder=webfolder, model=DefaultLLM)
    if update:
        print("Embeddings updated")
    else:
        print("Embeddings not updated")
    print("\n\nLoad Embeddings and show metadata")
    (embeddings, metadata)=load_embeddings(webfolder="short")
    showmetadata(metadata)
    print("\n%d embeddings loaded" % len(embeddings))
    #
    # Start the interactive loop
    # Log the interactions to a file
    logfile="data/"+webfolder+"_log.md"
    f = open(logfile, "w")
    while True:
        print("\n\nEnter your question or type 'quit' to exit:")
        question=input()
        if question=="quit":
            break
        f.write("**Q:** "+question+"\n")
        # Get the embedding for the question
        ind=getsimilar(embeddings, question)
        print(ind)
        # Get the top 3 texts
        text=gettopN(embeddings, metadata, ind, N=3)

        # Write the text to the logfile
        f.write("**A:** "+text+"\n")
        # Print the text
        print(text)
    print("Bye!")

if __name__ == "__main__":
    main()
