# pyinstaller --onedir --hidden-import PySide6 --hidden-import langchain --hidden-import openai --hidden-import bldgGen --hidden-import encGen --hidden-import grpGen --hidden-import npcGen --hidden-import pwEngine --hidden-import twnGen --hidden-import ui_form --hidden-import worldManager mainwindow.py
# python setup.py build

import sys
import os
from cx_Freeze import setup, Executable

# Define the list of files you want to include (besides your script).
include_files = ["wildspace.ico"]

# Create an Executable object for your script.
target = Executable(
    script="mainwindow.py",  # Replace with the name of your script.
    base="Win32GUI",  # Change to "Win32GUI" for a GUI application on Windows.
    icon="wildspace.ico",  # Include the path to an icon file if needed.
)

# Define the modules and packages you want to include.
# Make sure to include PySide6 and its submodules.
packages = ["PySide6", "langchain", "openai"]

# Create the setup call.
setup(
    name="Project Wildspace",  # Change to your project name.
    version="0.0.1",
    description="Generative AI Worldbuilding Assistant",
    options={'build_exe': {'include_files': include_files, 'packages': packages}},
    executables=[target],
)
