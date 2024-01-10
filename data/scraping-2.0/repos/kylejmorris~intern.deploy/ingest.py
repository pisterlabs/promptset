import os
os.environ["OPENAI_API_KEY"] = "API here"

from langchain.llms import OpenAI
import git

llm = OpenAI(model_name="text-davinci-003", n=1, best_of=5,max_tokens=-1)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

embeddings = OpenAIEmbeddings()
vectorstore = Chroma("repo_embeddings", embeddings, persist_directory="repo_embeddings")
repos = [""" list of github repos here"""]

page_content = "lets see how this embeds"*100
id = vectorstore.add_documents([Document(page_content=page_content)])
import shutil

try:
    shutil.rmtree('repo')
except:
    pass
doc_to_repo = {}
id_to_repo = {}
for repo in repos:
    print("Handling repo: ", repo)

    _ = git.Repo.clone_from(repo,'repo')

    list_of_files = []
    for filename in os.listdir('repo'):
        f = os.path.join('repo', filename)
        # checking if it is a file
        if os.path.isfile(f):
            if f.endswith((".mod",".sum",".gitignore",".md",".sh","Dockerfile")):
                continue
            list_of_files.append(f)


    all_desc = ""
    for code_file in list_of_files:
        f = open(code_file, "r")
        code = f.read()
        f.close()
        prefix = "Describe what the following code does in detail-\n"
        #print(llm(prefix+code))
        try:
            all_desc += llm(prefix+code)

        except:
            #Hack to deal with long files
            if len(code)%2 == 0:
                code1 = code[0:len(code)//2]
                code2 = code[len(code)//2:]
            else:
                code1 = code[0:(len(code)//2+1)]
                code2 = code[(len(code)//2+1):]

            desc1 = llm(prefix+code1)
            desc2 = llm(prefix+code2)

            all_desc += desc1 + desc2




    repo_summary = "Individual code file in a repo perform following tasks, what does the entire service do in detail?\n" + all_desc
    print(llm(repo_summary))
    id = vectorstore.add_documents([Document(page_content=all_desc)])[0]
    doc_to_repo[all_desc] = repo
    id_to_repo[id] = repo
    shutil.rmtree('repo')

print("ids to repo: ", id_to_repo)
vectorstore.persist()
res = vectorstore.similarity_search("my build is stuck/suddenly slow",k=1)
print(f"Bug seems to be from {doc_to_repo[res[0].page_content]}")