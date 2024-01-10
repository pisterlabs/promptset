from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import FileResponse
from app.utils.assistant_utils import delete_project_assistants_util
from app.utils.chat_utils import delete_project_chats_util
from app.models.pydantic_models import Project, InitializeProjectRequest, UpdateProgressRequest
from app.models.database import projects
from app.dependencies import get_database
from app.core.config import settings
from openai import OpenAI
from sqlalchemy import desc
from typing import List
import zipfile
import datetime
import shutil
import os

router = APIRouter()
client = OpenAI(api_key=settings.openai_api_key)
database = get_database()

@router.get("", response_model=List[Project])
async def get_all_projects():
    try:
        # Query to select all projects ordered by creation date in descending order
        query = projects.select().order_by(desc(projects.c.created_at))
        results = await database.fetch_all(query)
        
        # Transform the database results into a list of Project instances
        projects_list = [Project(**result) for result in results]

        return projects_list
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("")
async def initialize_project(request_body: InitializeProjectRequest):
    try:
        # Generate the current timestamp without microseconds
        timestamp = datetime.datetime.now()

        # Modify timestamp and name string for filesystem-friendly folder naming
        folder_timestamp = timestamp.strftime('%Y_%m_%d_at_%H_%M_%S')
        folder_name = request_body.name.replace(' ', '_').lower()
        project_folder = f"genraft_ai_data/projects/{folder_name}_{folder_timestamp}"
        os.makedirs(project_folder, exist_ok=True)

        # Write the initial idea into a text file
        with open(f"{project_folder}/prompt.txt", 'w') as file:
            file.write(request_body.idea)

        # Insert the project with the folder path and datetime object
        query = projects.insert().values(
            name=request_body.name,
            idea_initial=request_body.idea,
            idea_final="",
            technical_plan="",
            folder_path=project_folder,
            created_at=timestamp
        )

        # Perform the database operation within a transaction
        async with database.transaction():
            new_project_id = await database.execute(query)

        return {"id": new_project_id}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{id}", response_model=Project)
async def get_project_by_id(id: int = Path(..., description="The ID of the project to retrieve")):
    try:
        # Existing logic to fetch project
        select_query = projects.select().where(projects.c.id == id)
        project = await database.fetch_one(select_query)

        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Convert the database model to the Pydantic model
        return Project(**project)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{id}")
async def delete_project_by_id(id: int = Path(..., description="The ID of the project to delete")):
    try:
        # Start by checking if the project exists
        select_query = projects.select().where(projects.c.id == id)
        existing_project = await database.fetch_one(select_query)

        if existing_project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get the folder path
        folder_path = existing_project['folder_path']

        # If the project exists, proceed to delete it
        delete_query = projects.delete().where(projects.c.id == id)
        await database.execute(delete_query)

        # Delete all chats from api and db entries associated with project id
        await delete_project_chats_util(project_id=id)
        
        # Delete all assistants from api and db entries associated with project id
        await delete_project_assistants_util(project_id=id)

        # Delete the associated folder
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        # Also delete the associated ZIP file if it exists
        zip_path = f"{folder_path}.zip"
        if os.path.exists(zip_path):
            os.remove(zip_path)

        return {"message": "Project successfully deleted"}
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{id}/download")
async def download_project_by_id(id: int = Path(..., description="The ID of the project to download")):
    try:
        # Check if the project exists and get its folder path
        select_query = projects.select().where(projects.c.id == id)
        existing_project = await database.fetch_one(select_query)

        if existing_project is None:
            raise HTTPException(status_code=404, detail="Project not found")

        folder_path = existing_project['folder_path']

        # Create a ZIP file from the folder
        zip_path = f"{folder_path}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    zipf.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), 
                               os.path.join(folder_path, '..')))

        # Return the ZIP file as a response
        return FileResponse(zip_path, media_type='application/octet-stream', filename=os.path.basename(zip_path))
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{id}/update-progress")
async def update_project_progress(id: int, request_body: UpdateProgressRequest):
    try:
        update_query = projects.update().\
            where(projects.c.id == id).\
            values(current_phase=request_body.current_phase, current_stage=request_body.current_stage)
        await database.execute(update_query)
        return {"message": "Project updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))