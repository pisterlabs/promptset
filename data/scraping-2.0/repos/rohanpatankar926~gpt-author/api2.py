# Example usage:
from dep2 import *
from fastapi import FastAPI 
from pydantic import BaseModel
import uvicorn
import openai
openai.api_key = "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" # get it at https://platform.openai.com/
stability_api_key = "sk-ATRFzLsqx4GmE1Yr3ddQGNa1ay8vHUl3TOO4auxRw9ktmwgm" # get it at https://beta.dreamstudio.ai/

app=FastAPI()



@app.post("/generate")
async def gpt_author_api(prompt:str,num_chapters:int,writing_style:str):
    novel, title, chapters, chapter_titles = write_fantasy_novel(prompt, num_chapters, writing_style)
    # Replace chapter descriptions with body text in chapter_titles
    for i, chapter in enumerate(chapters):
        chapter_number_and_title = list(chapter_titles[i].keys())[0]
        chapter_titles[i] = {chapter_number_and_title: chapter}
    # Create the cover
    create_cover_image(str(chapter_titles))
    # Create the EPUB file
    create_epub(title, "GPT-Author", chapter_titles, 'cover.png')
    print(title)
    return {"title":title,"chapter_titles":chapter_titles}
if __name__=="__main__":
    uvicorn.run(app=app,port=3453,host="0.0.0.0")