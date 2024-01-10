import flet as ft 
import openai
from openai_client import OpenAIClient 
import os 

client = OpenAIClient() 
openai.api_key = client.api_key 

def process_file_contents(file_contents: str):
    goals = """Develop a treatment plan for the patient below using this format:
        1. Specific tissue diagnosis and stage
        2. Goals of treatment
        3. Initial treatment plan and proposed duration
        4. Expected common and rare toxicities during treatment and their management
        5. Expected long-term effects of treatment
        6. Psychosocial and supportive care plans
        7. Advanced care directives and preferences
        """

    messages = [
        {"role": "user", "content": goals},
        {"role": "user", "content": file_contents}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=700,
        temperature=0.75,
        messages=messages
    )
    return response.choices[0].message.content

def main(page: ft.Page):
    page.theme_mode = ft.ThemeMode.DARK
    page.scroll = "always"
    page.fonts = { "Inter" : "https://github.com/rsms/inter/raw/master/docs/font-files/InterDisplay.var.ttf" } 

    def pick_files_result(e: ft.FilePickerResultEvent):

        if e.files is not None: 
            file_path = os.path.join(os.path.expanduser("~/Desktop"), e.files[0].name)
            with open(file_path, 'r') as file: 
                file_contents = file.read() 
            processed_content = process_file_contents(file_contents)
            chat.controls.append(ft.Text(f"{processed_content}"))
            page.update()

        selected_files.value = (
            ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
        )
        selected_files.update()

    chat = ft.Column()
    file_picker = ft.FilePicker(on_result=pick_files_result)
    selected_files = ft.Text()
    page.overlay.append(file_picker)

    page.add(
        ft.AppBar(title=ft.Text("STEM Fellowship Project", size=30, font_family="Inter"), bgcolor="#1B2430"),
        chat,
        ft.Row(
            [
                ft.ElevatedButton(
                    "Upload patient information (.txt)",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: file_picker.pick_files(
                        allow_multiple=False
                    ),
                ),
                selected_files,
            ]
        )
    )

ft.app(target=main, view=ft.WEB_BROWSER)