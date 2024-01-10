import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("FeaturePipeline")
   image = modal.Image.debian_slim().pip_install(["hopsworks","pydrive","pyPDF2","openai","pandas"])

   @stub.function(image=image, schedule=modal.Period(days=1),mounts=[modal.Mount.from_local_dir("/Users/jiarro/BYOC-Build-Your-Own-Scalable-Chatbots/FeaturePipeline", remote_path="/root")],secret=modal.Secret.from_name("features"))
   def f():
       g()





def g():
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    import os
    import hopsworks
    import PyPDF2
    import openai
    from openai import OpenAI
    from math import ceil
    import json
    import pandas as pd


    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    
    project = hopsworks.login(api_key_value=os.environ["HOPSWORKS"])
    fs = project.get_feature_store()
    print(os.listdir())
    #AUTH in GDrive
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile("credentials.json")

    drive = GoogleDrive(gauth)

    #GET FOLDER ID FROM ENV
    folder_id = os.environ["FOLDER_ID"]


    #READ EMBEDDINGS
    embeddings_fg = fs.get_feature_group(name="embeddings",version=1)
    instructions_fg = fs.get_feature_group(name="instructionset",version=4)
    embeddings_df = embeddings_fg.read()


    #READ FROM GDrive FOLDER
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList() 
    new_files = []
    e_set = set(embeddings_df["source"])

    #GET ONLY FILES NOT ELABORATED BEFORE
    for file in file_list:
        if file["title"] not in e_set: 
            file.GetContentFile(file["title"])
            new_files.append(file)

    transcriptions = [] 
    embeddings = {}

    #GET TRANSCRIPTION OF NEW FILES
    for file in new_files:
        pdfReader = PyPDF2.PdfReader(file["title"])

        count = len(pdfReader.pages)
        output = ""
        embeddings[file["title"]] = {"text":[]}
        for i in range(count):
            pageObj = pdfReader.pages[i]
        
            extr = pageObj.extract_text()
            embeddings[file["title"]]["text"].append(extr)
            output += "\n" + extr
            
        transcriptions.append(output)

    responses = []

    #CHUNKS THEM AND ASK GPT FOR BUILDING THE INSTRUCTION SET
    for t in transcriptions:
        context = t
        i = 0
        chunks = []

        for i in range(ceil(len(t)/4097)):
            chunks.append(t[i*4097:i*4097+4097])

        for c in chunks:
            context = c
            question = "The text above is the result of the transcription of slides in the PDF file format. Remove chapter names and slides numbers and rephrase the sentences. Once you do that generate 2 to 3 meaningful questions on the text and the respective answers. Plese reply in the JSON format {'questions':<questions generated>,'answers':<answers generated>}. DO NOT write anything else than the requested JSON and remember to write the full elaborated content and not just one part."
            #question = "The text above is the result of the transcription of slides in the PDF file format. Remove chapter names and slides numbers and rephrase the sentences. Once you do that generate 3 meaningful questions based on the new text and the respective answers. As for the reply, follow the following template FOR EACH pair of question and the respective answer: '[INST] <question> [/INST] <answer>'  and so on, let's call this template a 'block'.  NEVER use newlines other than separating blocks and NEVER write anything that is not formatted as the proposed template. DO NOT write anything else than the requested blocks and make sure everything is formatted correctly."
            # response = openai.Completion.create(
            # engine="gpt-3.5-turbo",
            prompt=f"\nContext: {context}\nQuestion: {question}"
            # )
            # answer = response.choices[0].text.strip()
            # print(answer)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            
            #print(response.choices[0].message.content)
            responses.append(response.choices[0].message.content)

    row_result = {"prompt":[],"questions":[],"answers":[]}
    # BUILD INSTRUCTION SET
    for i, r in enumerate(responses):
        try:
            tmp = json.loads(r)
            for j in range(len(tmp["questions"])):
                instr = f"<s> [INST] {tmp['questions'][j]} [/INST] {tmp['answers'][j]} </s>"
                row_result["questions"].append(tmp['questions'][j])
                row_result["answers"].append(tmp['answers'][j])
                row_result["prompt"].append(instr)
                # print(instr)
        except:
            pass

    instructions = pd.DataFrame(row_result)
    #pd.set_option('display.max_colwidth', None)

    #BUILD EMBEDDING FG

    emb = {"source":[],"page":[],"content":[]}
    for e in embeddings:
        for idx,t in enumerate(embeddings[e]["text"]):
            emb["source"].append(e)
            emb["page"].append(idx)
            emb["content"].append(t)



    embedding_df = pd.DataFrame(emb)

    if not embedding_df.empty:
        embeddings_fg.insert(embedding_df)
    if not instructions.empty:
        instructions_fg.insert(instructions)











    

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("FeaturePipeline")
        with stub.run():
            f()