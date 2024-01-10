from database import db_cookings
from fastapi import APIRouter, Depends, HTTPException, Body
from models.cooking import Cooking, NewCooking, Issue
from routes.auth import User, get_current_active_user
from services.openai_service import OpenAIService
from services.recipe_service import SpoonacularAPIService

router = APIRouter(prefix="/cookings", tags=["cookings"])


@router.post("/", response_model=Cooking)
async def new_cooking(
        cooking: NewCooking,
        user: User = Depends(get_current_active_user),
):
    recipe = await SpoonacularAPIService.get_full_recipe(cooking.recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    cooking = Cooking(recipe=recipe)

    db_cookings[user.id] = cooking

    return cooking


@router.patch("/step/{step}", response_model=Cooking)
def set_step(
        step: int,
        user: User = Depends(get_current_active_user),
):
    if user.id not in db_cookings:
        raise HTTPException(status_code=404, detail="No cooking in progress")
    cooking = db_cookings[user.id]

    if step < 0 or step >= len(cooking.recipe.instructions):
        raise HTTPException(status_code=400, detail="Invalid step")
    cooking.current_step = step

    return cooking


@router.get("/", response_model=Cooking)
def get_cooking(
        user: User = Depends(get_current_active_user),
):
    if user.id not in db_cookings:
        raise HTTPException(status_code=404, detail="No cooking in progress")
    return db_cookings[user.id]


@router.delete("/")
def delete_cooking(
        user: User = Depends(get_current_active_user),
):
    if user.id not in db_cookings:
        raise HTTPException(status_code=404, detail="No cooking in progress")
    del db_cookings[user.id]
    return {"message": "Cooking stopped"}


@router.post("/issues", response_model=Issue)
def get_help(
        issue: str = Body(...),
        user: User = Depends(get_current_active_user),
):
    if user.id not in db_cookings:
        raise HTTPException(status_code=404, detail="No cooking in progress")
    cooking = db_cookings[user.id]

    response = OpenAIService.get_help(cooking, issue)
    cooking.issues.append(
        Issue(
            description=issue,
            step=cooking.current_step,
            solution=response,
        )
    )

    return cooking.issues[-1]
