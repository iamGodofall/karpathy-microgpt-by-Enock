"""
Setup script for microgpt package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="microgpt",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A minimal GPT implementation in pure Python with full training infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/microgpt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
    ],
    extras_require={
        "viz": ["plotly>=5.0", "matplotlib>=3.5"],
        "web": ["flask>=2.0"],
        "dev": ["pytest>=7.0", "black>=22.0", "flake8>=5.0"],
        "all": ["plotly>=5.0", "matplotlib>=3.5", "flask>=2.0", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "microgpt=cli:main",
        ],
    },
)
