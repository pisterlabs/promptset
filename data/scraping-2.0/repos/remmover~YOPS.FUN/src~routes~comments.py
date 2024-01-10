from fastapi import APIRouter, Depends, HTTPException, Path
import openai
from sqlalchemy.ext.asyncio import AsyncSession
import re


from src.database.connect import get_db
from src.database.models import Comment, Image, User
from src.schemas import (
    ReturnMessageResponseSchema,
    CommentDb,
    CommentShowAllSchema,
    CommentUpdateSchema,
)
from starlette.background import BackgroundTasks


from src.conf.config import config
from src.services.auth import auth_service
from src.repository import comments as repository_comments

router = APIRouter(prefix="/comments", tags=["comments"])

openai.api_key = config.gpt_api_key


emo_joy = re.compile(r"^.*?<\(@.*?joy.+?(\d{1,3})\s*\%", re.I|re.M)
emo_anger = re.compile(r"^.*?<\(@.*?anger.+?(\d{1,3})\s*\%", re.I|re.M)
emo_sadness = re.compile(r"^.*?<\(@.*?sadness.+?(\d{1,3})\s*\%", re.I|re.M)
emo_surprise = re.compile(r"^.*?<\(@.*?surprise.+?(\d{1,3})\s*\%", re.I|re.M)
emo_disgust = re.compile(r"^.*?<\(@.*?disgust.+?(\d{1,3})\s*\%", re.I|re.M)
emo_fear = re.compile(r"^.*?<\(@.*?fear.+?(\d{1,3})\s*\%", re.I|re.M)

async def emometer(comment: str, comment_id: int, db: AsyncSession):
    template = "Imagine that you show emotions of joy, anger, sadness, " \
        "surprise, disgust, and fear of people in messages which are " \
        "quoted with \"&&&\" sequence. Rate the following messages in " \
        "percentage of each emotion. Each rate must start with \"<(@\" "\
        "sequence and end with \"@)>\". Here is the list of my messages:"
    comments = [comment]
    request = ""
    i = 1
    for com in comments:
        request += "\n" + f"{i}. &&&{com}&&&."
        i += 1
    request = template + request
    # models = openai.Model.list()
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request}])
    answ: str = chat_completion.choices[0].message.content

    joy, anger, sadness, surprise, disgust, fear = 0, 0, 0, 0, 0, 0

    m = emo_joy.search(answ)
    if m:
        joy = int(m.group(1))
    m = emo_anger.search(answ)
    if m:
        anger = int(m.group(1))
    m = emo_sadness.search(answ)
    if m:
        sadness = int(m.group(1))
    m = emo_surprise.search(answ)
    if m:
        surprise = int(m.group(1))
    m = emo_disgust.search(answ)
    if m:
        disgust = int(m.group(1))
    m = emo_fear.search(answ)
    if m:
        fear = int(m.group(1))
    await repository_comments.update_emometer(
        joy, anger, sadness, surprise, disgust, fear,
        comment_id, db
    )


@router.post("/", response_model=CommentDb)
async def create_comment_for_image(
    image_id: int,
    comment: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(
            status_code=404,
            detail="Image not found or doesn't belong to the current user",
        )

    new_comment: Comment = await repository_comments.create_comment(
        comment, current_user.id, image_id, db
    )

    if new_comment:
        background_tasks.add_task(emometer, comment, new_comment.id, db)

    return new_comment


@router.put("/", response_model=CommentUpdateSchema)
async def update_comment_for_image(
    body: CommentUpdateSchema,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    updated_comment = await repository_comments.update_comment(body, current_user, db)

    if updated_comment:
        background_tasks.add_task(emometer,
                                  updated_comment.comment,
                                  updated_comment.id, db)
        return {
            "comment_id": updated_comment.id,
            "image_id": updated_comment.image_id,
            "comment": updated_comment.comment,
            "message": "Comment description is successfully changed.",
        }

    raise HTTPException(
        status_code=404,
        detail="Comment not found or doesn't belong to the current user",
    )


@router.get("/images/{image_id}/comments/", response_model=CommentShowAllSchema)
async def get_comments_for_image(image_id: int, db: AsyncSession = Depends(get_db)):
    comments = await repository_comments.get_comments_for_image(image_id, db)
    if comments:
        return {"comments": comments}

    raise HTTPException(
        status_code=404,
        detail="Comment not found or doesn't belong to the current user",
    )


@router.delete("/{comment_id}", response_model=ReturnMessageResponseSchema)
async def delete_comment_for_image(
    comment_id: int = Path(description="The ID of comment to delete", ge=1),
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    deleted_comment = await repository_comments.delete_comment(
        comment_id, current_user, db
    )

    if deleted_comment:
        return {"message": f"Comment with ID {comment_id} is successfully deleted."}
    else:
        raise HTTPException(
            status_code=404,
            detail="Comment not found or doesn't belong to the current user",
        )
