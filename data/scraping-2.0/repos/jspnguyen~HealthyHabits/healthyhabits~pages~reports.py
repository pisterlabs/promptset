"""The reports page. This file includes examples abstracting complex UI into smaller components."""
import reflex as rx
from healthyhabits.state.base import State
from healthyhabits.state.home import HomeState
from healthyhabits.templates import template
from healthyhabits import styles
#I imported json here, wasn't there originally
import json
import openai


from ..components import container



@template(route="/reports", title="Reports", image="/reports.png")
def reports() -> rx.Component:
    """The reports page.

    Returns:
        The UI for the reports page.
    """
    header = rx.text(
                "Reports",
                background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                background_clip="text",
                font_weight="bold",
                font_size="4em", 
                text_align = "center", 
               )
             
          


    return rx.vstack(
        header,
        rx.divider(),
        # ADDED FOR EACH LOOP FOR REPORTING REPORTS 
        rx.foreach(
            HomeState.sessions,
            lambda text: rx.box(
                rx.hstack(
                    rx.text(f"Most Common Emotion: {text['max_emotion']}", font_size="0.8em",background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                    background_clip="text", text_align="center",
                    font_weight="bold", width = "20vh"),
                    rx.text(f"Start of Session: {text['start_time']}", font_size="0.8em",background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                    background_clip="text", text_align="center",
                    font_weight="bold", width = "20vh"),
                    rx.text(f"End of Session: {text['end_time']}", font_size="0.8em",background_image="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                    background_clip="text", text_align="center",
                    font_weight="bold", width = "20vh"),
                    rx.button("Email Raw Data to Self", on_click=HomeState.email(HomeState.form_data["email"],text), bg="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                color="white",
                _hover={"shadow": "0 4px 60px 0 rgba(0, 0, 0, 0.3), 0 4px 16px 0 rgba(0, 0, 0, 0.3)"}, width = "25vh"),
                    rx.button("Email Report to Self", on_click=HomeState.emailFull(HomeState.form_data["email"],text), bg="linear-gradient(271.68deg, #EE756A 0.75%, #756AEE 88.52%)",
                color="white",
                _hover={"shadow": "0 4px 60px 0 rgba(0, 0, 0, 0.3), 0 4px 16px 0 rgba(0, 0, 0, 0.3)"}, width = "25vh"),
                spacing="1em",
            ),
        ),
        ),
    )