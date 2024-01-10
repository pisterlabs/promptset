import chunk
import os
import time
from roundrobin import RoundRobin
import openai
import subprocess
from multiprocessing import Pool, Process, Manager

rr = RoundRobin(['OAI1',
                 'OAI2',
                 'OAI3',
                 'OAI4',
                 'OAI5'])

def gpt3(prompt, tokens=600):

    # get the system variable for rr.next()
    key = os.environ.get(rr.next())

    openai.api_key = key
    
    return openai.Completion.create(
    model="text-davinci-002", 
    prompt=prompt, 
    temperature=0.3, 
    max_tokens=tokens)['choices'][0]['text']

chunk_size=1500

p1 = """

We need you to perform the following modifications to this document, outputting the modified document in full afterwards:

* Remove anything from the text above which doesn't look like natural language, smooth it out.
* Words which are wrapped on the next line and hyphenated i.e. morpho-
phonology should become "morphophonology"
* All references to figures or tables should be removed
* All parenthesized text should be removed, e.g. "(Solar-Lezama et al., 2016)"
* Cover the entire input document above then terminate
* remove all unnecessary whitespace and carriage returns
* Only write responses in natural explanatory language, remove all mathematics, notations, symbols etc
* Remove any empty lines, or lines which only have a few words or symbols on them

This is the what the document will look like after you have completed the task:
    """

p2 = """

Provide an expert summary of the document in natural language, explaining the main points and summarizing the main ideas. 
You can use the document above as a reference, but you should not copy any of it. 
You should write your summary in natural language, not in the form of a list of bullet points."""

p3 = """

Take the text above and represent with a nested bullet tree structure (using this bullet: "*") with abstract very high-level short headings (use tabbed indentation):"""


# for each chunk of text, call the gpt3 function and pass the prompt and the number of tokens to generate

# create a pool of processes

# for each chunk of text, call the gpt3 function and pass the prompt and the number of tokens to generate
def process_chunk_base(item):

    (d, index, chunk) = item

    response1 = gpt3(chunk + p1, chunk_size )
    response2 = gpt3(response1 + p2, 800)

    d[index] = response2
    return index

def process_chunk_l1(item):

    (d, index, chunk) = item

    response2 = gpt3(chunk + p2, 800 )
    
    d[index] = response2
    return index

def process_pdf(url, html=False):

    # download this pdf and save it as paper.pdf
    os.system("wget " + url + " --user-agent TryToStopMeFromUsingWgetNow -O paper.pdf")
    result = subprocess.run(['pdf2txt.py', 'paper.pdf'], stdout=subprocess.PIPE)
    text = result.stdout.decode('utf-8')

    # chunk the text up into 1000 word chunks and store them in a list

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    manager = Manager()

    with Pool() as pool:

        l1_done = False

        # record the current time
        start_time = time.time()
        level = 0

        while(len(chunks)>1):

            d = manager.dict()
            fun = process_chunk_l1 if l1_done else process_chunk_base

            round_time = time.time()
            for result in pool.imap(fun, map(lambda i: (d, i[0], i[1]),enumerate(chunks)) ):
                print(f"Completed chunk {result} on level {level} in {time.time() - round_time} seconds")
                
            l1_done = True
            level += 1

            # concatenate all the adjacent chunks together i.e. chunks 0 and 1, 2 and 3 etc
            # we need a special case if the length is odd, if so just ignore the last odd chunk on this iteration
            if len(chunks)%2==0:
                chunks = [d[i]+d[i+1] for i in range(0, len(chunks), 2)]
            elif len(chunks) >1:
                # select the last chunk
                last_chunk = chunks[-1] 
                chunks = [d[i]+d[i+1] for i in range(0, len(chunks)-1, 2)]
                chunks.append(last_chunk)

            # print we have completed a level and say how many chunks we have left
            print(f"Completed level {level} in {time.time() - round_time} seconds, {len(chunks)} chunks remaining")

    # report how long it took to process the file
    print("--- %s seconds ---" % (time.time() - start_time))

    structure = gpt3(chunks[0] + p3, chunk_size )

    bullet_prompt = """

Take the text above and represent with an expert 15 (must be 15) bulletpoint summary, remove any references to \"The document discusses \":"""

    bullets = gpt3(chunks[0] + bullet_prompt, chunk_size )

    cleanup_prompt = """\n\nPlease cleanup how this document is formatted. remove any unnecessary whitespace, and remove any empty lines."""

    final_doc = structure + "\n\n" + bullets + "\n\n" + chunks[0]

    # print the final result
    print(final_doc)

    # if html is true, then we need to convert the final document to html
    if html:
        # take final_doc but encapsulate it in a small HTML document, place its contents in a textarea
        final_doc = f"""<html><head><title>Document</title></head><body><textarea rows="40" cols="100">{final_doc}</textarea></body></html>"""

    pool.close()

    return final_doc
