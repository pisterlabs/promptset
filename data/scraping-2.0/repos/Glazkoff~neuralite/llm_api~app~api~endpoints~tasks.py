from fastapi import APIRouter, HTTPException

from llm_api.app.models.schemas.task import (
    SummarizationTaskCreate,
    SummarizationTaskResponse,
)
from llm_api.app.models.domain.task import SummarizationTask
from llm_api.app.core.openai_service import OpenAIService

router = APIRouter()
openai_service = OpenAIService()


tasks = []


@router.get("/tasks/")
async def read_tasks():
    return tasks


@router.post("/tasks/", response_model=SummarizationTaskResponse)
def create_task(task: SummarizationTaskCreate):
    try:
        ai_summarization = openai_service.generate_summary(task.text)
        extracted_facts = openai_service.extract_facts(task.text)
        new_task = SummarizationTask(
            id=len(tasks),
            text=task.text,
            ai_summarization=ai_summarization,
            extracted_facts=extracted_facts,
        )
        tasks.append(new_task)
        return new_task
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise HTTPException(500, detail=error_message) from e
