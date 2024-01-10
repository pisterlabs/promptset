import asyncio
from datetime import datetime
import json
import os

from OpenAIAPI import OpenAIAPI
from PresentationParser import PresentationParser
from DB.database import engine, Upload
from sqlalchemy.orm import Session
from sqlalchemy import or_
from macros import MacrosStatus, OUTPUT_FOLDER

session = Session(engine)


async def process_file(file):
    print(f"Started processing file {file.uid}.pptx")

    # Parse presentation
    presentation_parser = PresentationParser(file.filename)
    slides = presentation_parser.process_presentation()

    # Generate explanations
    openai_api = OpenAIAPI(api_key="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")
    explanations = await openai_api.generate_explanations(slides)

    # Save explanations to file
    save_to_file(explanations, str(file.uid))
    print(f"Finished processing file {file.uid}.json")


def save_to_file(explanations, filename) -> None:
    output_file = os.path.join(OUTPUT_FOLDER, f"{filename}.json")
    output_data = {"explanations": explanations}
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)


async def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"Explainer system started. Monitoring db for new files...")

    while True:
        # Check for new files in Upload DB
        with Session(engine) as session:
            files = session.query(Upload).filter(or_(Upload.status == MacrosStatus.PENDING.value,
                                                     Upload.status == MacrosStatus.FAILED.value)).all()
            if not files:
                print("No new files found. Sleeping for 10 seconds...")
                await asyncio.sleep(10)
                continue

            for file in files:
                # file_path = os.path.join(file.filename, file.filename)
                if os.path.isfile(file.filename):
                    try:
                        await process_file(file)
                        file.status = MacrosStatus.DONE.value
                        file.finish_time = datetime.now()

                    except Exception as e:
                        print(f"Error processing file {file.filename}: {e}")
                        file.status = MacrosStatus.FAILED.value

                    session.commit()
            print("Sleeping for 10 seconds...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
