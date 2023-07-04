import ast
import re
from setuptools import setup, find_packages

_version_re = re.compile(r"__version__\s+=\s+(.*)")

name_pkg = "noolp"

with open(name_pkg + "/version.py", "rb") as f:
    VERSION = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

setup(
    name=name_pkg,
    version=VERSION,
    packages=find_packages(),
    url="",
    license="",
    author="Luca Giacomelli",
    author_email="lucagiacomelli1604@gmail.com",
    description="Natural Language Processing library",
    keywords=["pip"],
    install_requires=[
        "gensim>=4.1.2",
        "nltk>=3.6.6",
        "scikit-learn>=1.0.1",
        "torch>=2.0.0",
        "datasets>=2.13.0",
        "transformers>=4.30.2",
    ],
)
