from setuptools import setup, find_packages

setup(
    name="openai-compatible-memory-proxy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "requests",
        "pydantic"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An OpenAI-compatible proxy server with memory capabilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openai-compatible-memory-proxy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
