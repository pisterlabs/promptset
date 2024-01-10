from pathlib import Path
from typing import List

from langchain.schema import Document

import chainlit as cl

from hr_job_cv_matcher.model import CandidateProfile
from hr_job_cv_matcher.service.chart_generator import generate_chart


async def display_uploaded_job_description(application_doc: Document):
    application_path = Path(application_doc.metadata["source"])
    elements = [
        cl.Pdf(
            name=application_path.name,
            display="inline",
            path=str(application_path.absolute()),
        )
    ]
    await cl.Message(
        content=f"Processed: **{application_path.name}**",
        elements=elements,
    ).send()


async def render_barchart_image(ranking_message: str, sorted_candidate_profiles: List[CandidateProfile]):
    barchart_image = generate_chart(sorted_candidate_profiles)
    elements = [
        cl.Image(
            name="ranking_image1",
            display="inline",
            path=str(barchart_image.absolute()),
            size="large",
        )
    ]
    await cl.Message(content=ranking_message, indent=0, elements=elements).send()