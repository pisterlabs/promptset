from fastapi import FastAPI, HTTPException
import openai
from fastapi import APIRouter, Depends, HTTPException, Response
from ... import models, schemas,auth2
from ...database import get_db
from sqlalchemy.orm import Session
import json
from ...config import settings

# OpenAI API configuration
router = APIRouter(prefix="/story_writer", tags=["story_writer"])
openai.api_key = settings.OPENAI_API_KEY

@router.post("/get_story_title_and_chapter", response_model=list[schemas.StoryResponse])
async def get_story_title_and_chapter(
    request: schemas.StoryRequest, db: Session = Depends(get_db),current_user: int = Depends(auth2.get_current_user)
):
    try:
        description = request.description
        messages = [
            {
                "role": "system",
                "content": "you are professional ,your task is to guide a person regarding their queries.",
            },
            {
                "role": "user",
                "content": f"""write a title and list of chapters name by taking context from given description. Description-->{description}. Please provide response in JSON format. The format should be like this-->{'{{"title":"hello","chapters":["word","word","word"]}}'}. Make sure the chapters list only contains chapter names, not any numbers or 'chapter 1' like.   """,
            },
        ]
        responses = []

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        result = response.choices[0].message.content.strip()
        result = eval(result)  # Convert the string to a dictionary

        title = result.get("title", "")
        chapters = result.get("chapters", [])

        story = schemas.StoryResponse(title=title, chapters=chapters)
        responses.append(story)
        new_story = models.Story(story_title_chapter=result, story_till_now="",owner_id=current_user.id)
        db.add(new_story)
        db.commit()
        db.refresh(new_story)

        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_summary(text):
    try:
        messages = [
            {"role": "user", "content": f"generate summary of given text. text-->{text}"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        result = response.choices[0].message.content.strip()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_story_chapter", response_model=schemas.ChapterResponse)
async def get_story_chapter(request: schemas.ChapterRequest, db: Session = Depends(get_db),current_user: int = Depends(auth2.get_current_user)):
    try:
        chapter_info = request.description
        story_id = request.story_id
        chapter_name = request.chapter_name
        story = db.query(models.Story).filter(models.Story.id == story_id,models.Story.owner_id== current_user.id).first()
        story_info = json.dumps(story.story_title_chapter)
        story_info = f"""{story_info}"""

        previous_info = ""
        if story.story_till_now:
            previous_info = story.story_till_now
        print(previous_info)
        summary_till_now = get_summary(previous_info)
        messages = [
            {"role": "system", "content": story_info},
            {
                "role": "assistant",
                "content": f"""Summary till now -> {summary_till_now}  .Continue writing a new chapter {chapter_info} content by taking context from the above given chapter info and summary till now. Do not write chapter number in response content:""",
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        result = response.choices[0].message.content.strip()
        chapter_dict = {
            "chapter_name": chapter_name,
            "chapter_content": result
        }
        story_chapter_wise = story.story_chapter_wise if story.story_chapter_wise else []
        story_chapter_wise.append(chapter_dict)

        story_query = db.query(models.Story).filter(models.Story.id == story_id)
        story_query.update({"story_chapter_wise": story_chapter_wise, "story_till_now": previous_info + result})
        db.commit()
        content = result
        chapter = schemas.ChapterResponse(content=content)
        return chapter
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
