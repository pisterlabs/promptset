# from os.path import join, dirname, realpath
# import os
# import pytest
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select
# from uuid import uuid4

# import openai

# from backend.ai_experiments2 import chat_with_tools
# from backend import file
# from backend import models
# from backend import schemas

# directory = dirname(realpath(__file__))

# vin_paper_path = join(directory, "data", "s41586-022-05157-3.pdf")


# @pytest.fixture
# async def user_id(session):
#     user_id = str(uuid4())
#     session.add(models.Users(id=user_id))
#     await session.commit()
#     return user_id


# async def test_user(session: AsyncSession, user_id: str):
#     (await session.execute(select(models.Users.id).where(models.Users.id == user_id))).scalar_one()


# @pytest.fixture
# async def pdf_file_id(session: AsyncSession, user_id: str) -> int | None:
#     with open(vin_paper_path, "rb") as f:
#         pdf_data = f.read()

#     file_id = await file.process_synced_file(
#         session, pdf_data, user_id, mime_type="application/pdf"
#     )

#     return file_id


# async def test_pdf_file(session: AsyncSession, pdf_file_id: int | None):
#     text_content = (
#         (await session.execute(select(models.FileData).where(models.FileData.id == pdf_file_id)))
#         .scalar_one()
#         .text_content
#     )
#     assert text_content
#     assert "A microbial supply chain for production" in text_content


# async def test_chat(session: AsyncSession, user_id: str, pdf_file_id: int | None):
#     # TODO drop, only for testing
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     response, tokens = await chat_with_tools(
#         "Respond with the ID of a paper I have uploaded", session, user_id
#     )
#     print(response)
#     print(f"\n\n{tokens} tokens")
