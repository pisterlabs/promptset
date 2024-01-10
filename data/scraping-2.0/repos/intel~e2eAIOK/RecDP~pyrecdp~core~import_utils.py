"""
 Copyright 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import os
from typing import Optional
import pip
import importlib
import pathlib
import pkg_resources
from loguru import logger

def list_requirements(requirements_path):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires

def fix_package_name(package):
    a = package.split('>')[0]
    a = a.split('<')[0]
    b = a.split('=')[0]
    
    package_name_map = {
        'scikit-learn' : 'sklearn',
        'pyyaml' : 'yaml',
        'Faker' : 'faker',
        'python-docx': 'docx',
        'Pillow': 'pillow'
    }
    
    if b in package_name_map:
        b = package_name_map[b]
    #print(b)
    return b

def check_availability_and_install(package_or_list, verbose=1):
    def actual_func(package):
        pip_name = fix_package_name(package)
        try:
            return importlib.import_module(pip_name)
        except ImportError:
            pip.main(['install', '-q', package])
            #importlib.import_module(pip_name)
            
    if isinstance(package_or_list, list):
        if verbose == 1 and len(package_or_list) > 0:        
            logger.info(f"check_availability_and_install {package_or_list}")
        for pkg in package_or_list:
            actual_func(pkg)
    elif isinstance(package_or_list, str):
        if verbose == 1 and package_or_list != "":        
            logger.info(f"check_availability_and_install {package_or_list}")
        actual_func(package_or_list)
    else:
        raise ValueError(f"{package_or_list} with type of {type(package_or_list)} is not supported.")
        

def import_faiss(no_avx2: Optional[bool] = None, install_if_miss: bool = True):
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        if install_if_miss:
            os.system("pip install -q faiss-gpu")
            os.system("pip install -q faiss-cpu")
        else:
            raise ImportError(f"faiss package not found, "
                              f"please install it with 'pip install faiss-gpu && pip install faiss-cpu'")


def import_langchain(install_if_missing: bool = True):
    """
    Import langchain if available, otherwise raise error.
    """
    try:
        from langchain.document_loaders.base import BaseLoader
    except ImportError:
        if install_if_missing:
            os.system("pip install -q langchain")
        else:
            raise ImportError(f"langchain package not found, please install it with 'pip install langchain")


def import_sentence_transformers(install_if_missing: bool = True):
    """
    Import sentence_transformers if available, otherwise raise error.
    """
    try:
        import sentence_transformers

    except ImportError as exc:
        if install_if_missing:
            os.system("pip install -q sentence_transformers")
        else:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc


def import_unstructured(install_if_missing: bool = True):
    try:
        import unstructured  # noqa:F401
        from unstructured.partition.auto import partition
    except ImportError:
        if install_if_missing:
            os.system("pip install -q  unstructured")
            os.system("pip install -q  unstructured[ppt, pptx, xlsx]")
        else:
            raise ValueError(
                "unstructured package not found, please install it with "
                "`pip install unstructured`"
            )


def import_openai(install_if_missing: bool = True):
    try:
        import openai
    except ImportError:
        if install_if_missing:
            os.system("pip install -q openai")
        else:
            raise ValueError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )

def import_pysbd(install_if_missing: bool = True):
    try:
        import openai
    except ImportError:
        if install_if_missing:
            os.system("pip install -q pysbd")
        else:
            raise ValueError(
                "pysbd package not found, please install it with "
                "`pip install pysbd`"
            )


