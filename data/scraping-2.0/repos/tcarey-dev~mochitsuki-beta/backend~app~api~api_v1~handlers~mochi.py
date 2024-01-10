from fastapi import APIRouter, Depends, Body
from ...deps.user_deps import get_current_user
from ....services.mochi_service import MochiService
from ....services.openai_service import OpenAIService
from ....models.user_model import User


mochi_router = APIRouter()


@mochi_router.get('/list', summary="Get all the mochi decks of the user")
async def list_mochi(current_user: User = Depends(get_current_user)):
    """Retrieve a list of all users mochi decks"""
    return await MochiService.list_mochi()


@mochi_router.post('/create/deck', summary="Create a deck in Mochi")
async def create_deck(deck_in: str, parent_in: str = None, current_user: User = Depends(get_current_user)):
    return await MochiService.create_deck(deck_in, parent_in)


@mochi_router.post('/create/card', summary="Create a card in Mochi")
async def create_card(deck_in: str = Body(embed=True), parent_in: str = Body(None, embed=True), text_in: str = Body(embed=True), current_user: User = Depends(get_current_user)):
    gpt = await OpenAIService.get_flashcard(deck_in, text_in, parent_in)
    flashcards = gpt.strip().split('\n\n')
    for i, flashcard in enumerate(flashcards):
        question, answer = flashcard.split('\nA:\n')
        content = f"{question.strip()}\n---\n{answer.strip()}\n"
        await MochiService.create_card(deck_in, content)
    pass
