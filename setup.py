from setuptools import setup, find_packages

with open("PIPREADME.md", "r", encoding="utf-8") as fh:
    pip_description = fh.read()

setup(
    name="intelli",
    version="1.0.1",
    author="Intellinode",
    author_email="admin@intellinode.ai",
    description="Build AI agents and MCPs with Intellinode",
    long_description=pip_description,
    long_description_content_type="text/markdown",
    url="https://www.intellinode.ai/",
    project_urls={
        "Source Code": "https://github.com/intelligentnode/Intelli",
    },
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={"": ["*.in"]},
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=1.0.0",
        "networkx>=3.2.0",
        "numpy<2.0"
    ],
    extras_require={
        "visual": ["matplotlib>=3.6.0"],
        "offline": ["keras-nlp", "keras>=3", "librosa", "keras-hub", "tensorflow-text"],
        "llamacpp": ["llama-cpp-python>=0.3.7", "huggingface_hub>=0.28.1"],
        "mcp": ["mcp~=1.9.0"],
        "all": ["matplotlib>=3.6.0", "keras-nlp", "keras>=3", "librosa", "keras-hub", 
                "tensorflow-text", "llama-cpp-python>=0.3.7", "huggingface_hub>=0.28.1", 
                "mcp~=1.9.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
