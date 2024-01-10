from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "This repo requires Python 3.6 or greater." \
    + "Please install it before proceeding."

setup(
    name='robust_rmab',
    py_modules=['robust_rmab'],
    version='0.1.0',
    install_requires=[
        'numpy',
        'pandas',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'torch',
        'gym',
        'nashpy',
        'tqdm'
    ],
    description="Robust RMAB double oracle and Deep RL code. Adapted from OpenAI's SpinningUp repository",
)
