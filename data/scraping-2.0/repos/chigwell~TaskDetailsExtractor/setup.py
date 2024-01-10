from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='TaskDetailsExtractor',
    version='0.0.2',
    description='Extract detailed tasks for software project modification using GPT from OpenAI.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Eugene Evstafev',
    author_email='chigwel@gmail.com',
    url='https://github.com/chigwell/TaskDetailsExtractor',
    packages=find_packages(),
    install_requires=[
        'openai',  # Ensure you specify the version of openai package you are using
        'projectstructor>=0.0.1'  # Assuming you're using functionalities from ProjectStructoR
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
