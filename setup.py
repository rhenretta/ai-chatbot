from setuptools import setup, find_packages

setup(
    name="ai-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "redis",
        "openai",
        "python-dotenv",
    ],
) 