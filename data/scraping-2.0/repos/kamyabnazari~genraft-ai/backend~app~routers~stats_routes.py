from fastapi import APIRouter, HTTPException
from app.models.pydantic_models import ProjectStats
from app.models.database import projects
from app.dependencies import get_database
from app.core.config import settings
from openai import OpenAI
import os

router = APIRouter()
client = OpenAI(api_key=settings.openai_api_key)
database = get_database()

@router.get("/projects", response_model=ProjectStats)
async def get_project_statistics():
    try:
        # Get all projects
        query = projects.select()
        all_projects = await database.fetch_all(query)
        total_projects = len(all_projects)

        total_files = 0
        total_assets = 0

        # File extensions considered as assets
        asset_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']

        # Count files and assets in each project folder
        for project in all_projects:
            folder_path = project['folder_path']
            for root, dirs, files in os.walk(folder_path):
                total_files += len(files)
                total_assets += sum(1 for file in files if os.path.splitext(file)[1].lower() in asset_extensions)

        # Combine the statistics into a single response object
        statistics = {
            "total_projects": total_projects,
            "total_files": total_files,
            "total_assets": total_assets
        }

        return statistics
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))