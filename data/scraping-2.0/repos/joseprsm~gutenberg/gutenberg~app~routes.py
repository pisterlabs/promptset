from typing import List, Any, Dict

import openai

from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from gutenberg.app.db import get_db
from gutenberg.app import models
from gutenberg.app import schemas

router = APIRouter()


@router.post('/')
def predict(prompt: schemas.Prompt, db: Session = Depends(get_db)):

    db_prompt = models.Prompt(
        item_name=prompt.item_name,
        item_description=prompt.item_description,
        target_audience=prompt.target_audience,
        platform=prompt.platform
    )

    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)

    predictions: List[Dict[str, Any]] = openai.Completion.create(
        engine="text-davinci-001",
        prompt=generate_prompt(
            prompt.item_name,
            prompt.item_description,
            prompt.platform,
            prompt.target_audience),
        temperature=0.6, n=5, max_tokens=1000
    ).choices

    for pred in predictions:
        pred_text = pred['text']
        db_pred = models.Prediction(prompt_id=db_prompt.id, text=pred_text)
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)

    return {
        'choices': [
            pred['text'] for pred in predictions
        ]}


@router.get('/')
def get_predictions(db: Session = Depends(get_db), skip: int = 0, limit: int = 20):
    return db.query(models.Prediction).offset(skip).limit(limit).all()


def generate_prompt(name, description, platform, target_audience):
    return f"""
        Write an ad for the following product to run on {platform} aimed at {target_audience}:
        Product: {name}. {description}.
    """
