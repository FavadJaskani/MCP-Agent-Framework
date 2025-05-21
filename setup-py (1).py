from setuptools import setup, find_packages

setup(
    name="mcp-agent",
    version="0.1.0",
    description="Multi-agent Communication Protocol (MCP) Framework",
    author="AI Agent Developer",
    author_email="developer@example.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "pydantic>=1.8.0",
        "python-dotenv>=0.15.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)