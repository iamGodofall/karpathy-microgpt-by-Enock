"""
Setup script for microgpt.
Supports both modern (pyproject.toml) and legacy (setup.py) installation.
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read version from __init__.py
with open("__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "2.0.0"

setup(
    name="microgpt",
    version=version,
    author="microgpt Team",
    author_email="team@microgpt.ai",
    description="A comprehensive, pure-Python GPT ecosystem with state-of-the-art features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamGodofall/karpathy-microgpt-by-Enock",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "web": [
            "flask>=2.0.0",
            "gunicorn>=20.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "export": [
            "torch>=1.10.0",
            "onnx>=1.12.0",
            "transformers>=4.20.0",
        ],
        "all": [
            "flask>=2.0.0",
            "gunicorn>=20.0.0",
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "torch>=1.10.0",
            "onnx>=1.12.0",
            "transformers>=4.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "microgpt=main:main",
            "microgpt-train=cli:train_cli",
            "microgpt-generate=cli:generate_cli",
            "microgpt-chat=chat:main",
            "microgpt-server=api_server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
