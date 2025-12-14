from setuptools import setup, find_packages

with open("PIPREADME.md", "r", encoding="utf-8") as fh:
    pip_description = fh.read()

setup(
    name="intelli",
    version="1.3.4",
    author="Intellinode",
    author_email="admin@intellinode.ai",
    description="Build AI agents and MCPs with Intellinode.",
    long_description=pip_description,
    long_description_content_type="text/markdown",
    url="https://www.intellinode.ai/",
    project_urls={
        "Source Code": "https://github.com/intelligentnode/Intelli",
    },
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={"": ["*.in"]},
    python_requires=">=3.10",
    install_requires=[
        "python-dotenv>=1.0.0",
        "networkx>=3.2.0",
    ],
    extras_require={
        "visual": ["matplotlib>=3.6.0"],
        "offline": ["keras-nlp", "keras>=3", "librosa", "keras-hub", "tensorflow-text"],
        "llamacpp": ["llama-cpp-python>=0.3.7", "huggingface_hub>=0.28.1"],
        "mcp": ["mcp[ws,cli]~=1.9.0", "pandas"],
        "dataframe": ["pandas", "polars>=0.19.0"],
        "speech": [
            "speechmatics-batch",
            "speechmatics-rt",
            "websockets",
            "librosa",
            "soundfile",
            "numpy",
            "openai>=2.5.0",
        ],
        "azure-assistant": ["openai>=2.5.0"],
        "all": [
            "matplotlib>=3.6.0",
            "numpy>=1.26.0,<2.2.0",
            "keras-nlp",
            "keras>=3",
            "librosa",
            "keras-hub",
            "tensorflow-text",
            "llama-cpp-python>=0.3.7",
            "huggingface_hub>=0.28.1",
            "mcp[ws,cli]~=1.9.0",
            "pandas",
            "polars",
            "speechmatics-batch",
            "speechmatics-rt",
            "websockets",
            "librosa",
            "soundfile",
            "numpy",
            "openai>=2.0.0",
        ],
        "dev": ["pytest>=7.0.0"],
    },
)
