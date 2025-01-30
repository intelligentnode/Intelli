from setuptools import setup, find_packages

with open("PIPREADME.md", "r", encoding="utf-8") as fh:
    pip_description = fh.read()

setup(
    name="intelli",
    version="0.5.0",
    author="Intellinode",
    author_email="admin@intellinode.ai",
    description="Create your chatbot or AI agent using Intellinode. We make any model smarter.",
    long_description=pip_description,
    long_description_content_type="text/markdown",
    url="https://www.intellinode.ai/",
    project_urls={
        "Source Code": "https://github.com/intelligentnode/Intelli",
    },
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={ 
        '': ['*.in']
    },
    python_requires='>=3.6',
    install_requires=[
        "python-dotenv==1.0.1", "networkx==3.2.1",
    ],
    extras_require={
        "visual": ["matplotlib==3.6.0"],
        "offline": ["keras-nlp", "keras>=3", "librosa", "keras-hub", "tensorflow-text"],
    }
)