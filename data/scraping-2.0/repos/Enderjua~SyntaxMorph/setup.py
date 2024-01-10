#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


import os


here = os.path.abspath(os.path.dirname(__file__))


with open('README.rst') as readme_file:
    readme = readme_file.read()


setup(
    name="SyntaxMorph",
    version="1.0.5",
    author="Marijua",
    author_email="enderjua@gmail.com",
    description="SyntaxMorph is a Python module that enables code conversion between different programming languages",
    long_description="""SyntaxMorph
SyntaxMorph
========

SyntaxMorph is a module that aims to facilitate the conversion between programming languages by utilizing OpenAI.

-  Free software: GPLv3 license
-  Github: https://github.com/Enderjua/SyntaxMorph



Features
~~~~~~~~

-  Determining which programming language a given code belongs to.
-  Identifying the general structure of the given code.
-  Converting the given code to the desired programming language.
-  Aiming to collect a comprehensive dataset.
-  Eliminating the dependency on OpenAI.

Versions
========

1.0.5
~~~~~~~~
-  Folder error resolved and published

1.0.4
~~~~~~~~
-  Folder error resolved and published

1.0.3
~~~~~~~~
-  Folder error resolved and published

1.0.2
~~~~~~~~
-  Published.


Developer
~~~~~~~~~

-  Marijua @ ``enderjua gmail com``


Quick Tutorial
--------------


    import openai
   
    openai.api_key = "YOUR_API_KEY"

    from morph import formatCode
    from morph import columDetect
    from morph import languageDetect
    
    
Language Detection
~~~~~~~~~~~~~~~~~~




    code = " print('hello world') "
    languageDetection = languageDetect.languageDetect(code)
    print("Language Detected: "+languageDetection) # Python



    Language Detected: Python
    


Colum Detection
~~~~~~~~~~~~



    code = " def main(a, b, c):
    
           d = a+b+c
           print(d)

     main(5,7,9)"
     columDetection = columDetect.columDetect(code)
     print("Colum Detected: "+columDetection) # Function && Fonksiyon




    Colum Detected: Fonksiyon




    print(columDetect.columDetect(code))




    Function && Fonksiyon


Language translation
~~~~~~~~~~~~~~~~~~~~~~



    code = " print('hello world') "
    
    newCode = formatCode.formatDetected(languageDetection, code, 1, C++, columDetection)
    print(newCode)
    
    




    #include <iostream>

    int main() {
        std::cout << "Hello World!" << std::endl;
        return 0;
    }


Create a function for Flask API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

main.py:



    import openai
    openai.api_key = "YOUR_API_KEY"
    
    from morph import formatCode as f
    from morph import languageDetect as l
    from morph import columDetect as c
    
    def morphApi(code, lang):
       language = l.languageDetect(code)
       colum = c.columDetect(code)
       newCode = f.formatDetected(language, code, 1, lang, colum)
       return newCode
       
    # code = morphApi("print('hello')", "C++")
    # print(code)




    #include <iostream>

    int main() {
        std::cout << "Hello World!" << std::endl;
        return 0;
    }


Create a Flask API
~~~~~~~~~~~~~~~~~~~~



    from flask import Flask, jsonify
    from flask_cors import CORS
    from urllib.parse import unqoute
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/translateAPI/<string:language>/<path:code>', methods=['GET'])
    def translating(language2, code):
      from main import morphApi
      code = morphApi(code, language2)
      return code
      
    if __name__ = '__main__':
        app.run(debug=True)
    




    localhost:5000/translateAPI/C++/print('hello world')
    
    #include <iostream>

    int main() {
        std::cout << "Hello World!" << std::endl;
        return 0;
    }
    

Future
~~~~~~~~

-  We have set out on the process of training our own AI.
-  We will share our AI for free here as a result of the AI training.
-  We will ensure the independence of OpenAI.




 """,
    long_description_content_type='text/markdown',
    url='https://github.com/Enderjua/SyntaxMorph',
    packages=find_packages(),
    package_data={'morph': ['columDetect.py', 'formatCode.py', 'languageDetect.py', 'main.py', 'translate.py']},  # Bu satırı ekleyin
    license="GPLv3",
    zip_safe=False,
    keywords='morph, syntax, python, syntaxmorph, ai, machinelearning, change, codexchange',
    install_requires=["openai", "black"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
