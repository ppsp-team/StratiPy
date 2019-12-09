import os
from setuptools import setup

setup(
    name = "stratipy",
    version = "0.8",
    author = "Yang-Min KIM, Guillaume DUMAS",
    author_email = "yang-min.kim@pasteur.fr, guillaume.dumas@pasteur.fr",
    description = ("Patients stratification with Graph-regularized"
                   " Non-negative Matrix Factorization (GNMF) in Python."),
    license = "BSD",
    keywords = "bioinformatics graph network stratification",
    url = "https://github.com/GHFC/StratiPy",
    packages=['stratipy', 'test', 'data'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: BSD License",
    ],
)
