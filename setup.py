from setuptools import setup, find_packages

with open("PIPREADME.md", "r", encoding="utf-8") as fh:
    pip_description = fh.read()

setup(
    name="intelli",
    version="0.0.2",
    author="Intellinode",
    author_email="admin@intellinode.ai",
    description="Create chatbots and AI agent work flows. It allows to connect your data with multiple AI models like OpenAI, Gemini, and Mistral through a unified access layer.",
    long_description=pip_description,
    long_description_content_type="text/markdown",
    url="https://www.intellinode.ai/",
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "requests-mock==1.11.0", "python-dotenv==1.0.1"
    ],
)