from settings import get_settings
from src.llms.openai import OpenAIManager
from langchain.embeddings.openai import OpenAIEmbeddings
from fastapi import File, UploadFile
from PIL import Image, ImageDraw
from io import BytesIO
import uvicorn
import aiofiles
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from src.schema import RequestLLM
from fastapi.responses import StreamingResponse
from src.prompt_template import Templates

app_settings = get_settings()
app = FastAPI()


embeddings = OpenAIEmbeddings(openai_api_key=app_settings.OPENAI_API_KEY)  # type: ignore
openai_summarizing = OpenAIManager(prompt_template=Templates.summarize_prompt)
openai_qa = OpenAIManager(prompt_template=Templates.openai_prompt_template)


@app.post("/index_image/")
async def upload_image(image_file: UploadFile = File(None)):
    print(image_file.filename)
    out_image_name = image_file.filename
    out_image_path = f"{app_settings.temp_folder}/{out_image_name}"
    request_object_content = await image_file.read()
    photo = Image.open(BytesIO(request_object_content))
    # make the image editable
    drawing = ImageDraw.Draw(photo)
    black = 3
    # font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    drawing.text((0, 0), "text_to_water_mark", fill=black)  # , font=font)
    photo.save(out_image_path)
    OpenAIManager.index_file_from_path(
        file_path=out_image_path, embedding_function=embeddings
    )

    return FileResponse(out_image_path, media_type="image/jpeg")


@app.post("/index_file/")
async def upload_file(file_: UploadFile = File(None)):
    print(file_.filename)
    out_image_name = file_.filename
    out_image_path = f"{app_settings.temp_folder}/{out_image_name}"

    async with aiofiles.open(out_image_path, "wb") as out_file:
        request_object_content = await file_.read()
        await out_file.write(request_object_content)

    OpenAIManager.index_file_from_path(
        file_path=out_image_path, embedding_function=embeddings
    )

    return FileResponse(out_image_path, media_type="image/jpeg")


@app.post("/summarize/")
async def request_summary(request: RequestLLM):
    return StreamingResponse(
        openai_summarizing.run_qa_chain(request.message, request.message_id),
        status_code=200,
        media_type="application/x-ndjson",
    )


@app.post("/question_answering/")
async def request_answer(request: RequestLLM):
    # Begin a task that runs in the background.

    return StreamingResponse(
        openai_qa.run_qa_chain(request.message, request.message_id),
        status_code=200,
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    uvicorn.run("main_endpoints:app", port=8000)
