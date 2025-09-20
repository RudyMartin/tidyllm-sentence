#!/usr/bin/env python3
"""
TidyLLM-Sentence - Sentence Embeddings and Transformers
======================================================
"""

from setuptools import setup, find_packages

# Read version from pyproject.toml
def get_version():
    try:
        import toml
        with open('pyproject.toml', 'r') as f:
            data = toml.load(f)
            return data['project']['version']
    except:
        return '0.1.0'

# Read README if available
def get_long_description():
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return "Pure Python sentence embeddings - tidyllm verse component for transparent, educational NLP"

setup(
    name="tidyllm-sentence",
    version=get_version(),
    author="Rudy Martin",
    author_email="rudy@nextshiftconsulting.com",
    description="Pure Python sentence embeddings - tidyllm verse component for transparent, educational NLP",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/rudymartin/tidyllm-sentence",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            # Future CLI commands can go here
        ],
    },
    zip_safe=False,
    include_package_data=True,
)