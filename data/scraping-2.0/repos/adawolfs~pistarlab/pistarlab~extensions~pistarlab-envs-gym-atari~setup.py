from setuptools import setup, find_packages

setup(
    name="pistarlab-envs-gym-atari",
    version="0.0.1.dev0",
    author="piSTAR",
    author_email="pistar3.14@gmail.com",
    description="ATARI games from OpenAI's gym",
    long_description='This is a pistarlab extension',
    url="https://github.com/pistarlab/pistarlab/extensions",
    license='',
    install_requires=['gym>=0.17.1'],
    package_data={'pistarlab-envs-gym-atari': ['README.md']
      },
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=3.6',
)