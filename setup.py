import setuptools
from distutils.util import convert_path

NAME = "pokerstar"
AUTHOR = "Anton Normelius"
EMAIL = "a.normelius@gmail.com"
DESCRIPTION = "World's best pokerbot"
URL = "https://github.com/normelius/PokerStar"
PACKAGES = ['pokerstar']
PYTHON_REQUIRES = ">=3.8"


with open("README.md", "r") as fh:
    long_description = fh.read()

# Read requirements.
with open('requirements.txt') as f:
    required = f.read().splitlines()
    print(required)

setuptools.setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    install_requires=required,
    packages=PACKAGES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=PYTHON_REQUIRES,
)
