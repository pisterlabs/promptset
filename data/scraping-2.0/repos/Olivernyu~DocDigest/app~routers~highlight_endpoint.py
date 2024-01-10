from fastapi import APIRouter, HTTPException, Query
from app.services.openai import OpenAIService
from ..shared_resources import page_data_store

router = APIRouter()


@router.get("/{page_id}")
async def highlight_page(
    page_id: str,
    ai_provider: str = Query("openai", description="AI provider for highlighting"),
):
    """
    Highlight the page by providing the page_id and an optional AI provider

    Parameters:
    - page_id (str): The ID of the page to be highlighted.
    - ai_provider (str): The AI provider for highlighting. Default is "openai".

    Returns:
    - dict: A dictionary containing the highlighted page html.
    """
    page_data = page_data_store.get(page_id)
    if page_data is None:
        raise HTTPException(status_code=404, detail="Page not found")
    text_from_html = page_data.get("text_from_html")
    if text_from_html is None:
        raise Exception("Text from html is not in page_data.")

    if ai_provider == "openai":
        openai_service = OpenAIService()
        highlighted_html = openai_service.highlight_text(text_from_html)
    else:
        raise HTTPException(status_code=400, detail="Unsupported AI provider")

    return {"highlighted_html": highlighted_html}
