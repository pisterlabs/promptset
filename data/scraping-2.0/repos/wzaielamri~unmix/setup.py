import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="unmix",
    py_modules=["unmix"],
    version="1.0",
    description="original code from OpenAI Jukebox",
    author="",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True
)
