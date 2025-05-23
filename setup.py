"""
Digital Twin Type 1 Diabetes Library
=====================================

A comprehensive digital twin library for Type 1 diabetes management
using state-of-the-art AI and machine learning techniques.
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
try:
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = __doc__

setup(
    name="digital-twin-t1d",
    version="2.0.0",
    author="Digital Twin T1D Consortium",
    author_email="info@digital-twin-t1d.org",
    description="State-of-the-art digital twin library for Type 1 diabetes management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/digital-twin-t1d/digital-twin-library",
    project_urls={
        "Bug Reports": "https://github.com/digital-twin-t1d/digital-twin-library/issues",
        "Source": "https://github.com/digital-twin-t1d/digital-twin-library",
        "Documentation": "https://digital-twin-t1d.readthedocs.io/",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies (always required)
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "torch>=1.10.0",
        "statsmodels>=0.13.0",
        "tqdm>=4.64.0",
        "joblib>=1.1.0",
    ],
    extras_require={
        "full": requirements,  # All dependencies
        "advanced": [
            "transformers>=4.20.0",
            "mamba-ssm>=1.0.0",
            "torchdiffeq>=0.2.3",
        ],
        "rl": [
            "stable-baselines3>=1.6.0",
            "gymnasium>=0.26.0",
            "gym>=0.21.0",
        ],
        "optimization": [
            "pymoo>=0.6.0",
            "optuna>=3.0.0",
            "econml>=0.14.0",
        ],
        "visualization": [
            "plotly>=5.0.0",
            "wandb>=0.13.0",
            "streamlit>=1.12.0",
            "dash>=2.6.0",
        ],
        "medical": [
            "simglucose>=0.2.1",
            "pydicom>=2.3.0",
            "hl7>=0.3.12",
        ],
        "deployment": [
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "redis>=4.0.0",
            "celery>=5.2.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "jupyter>=1.0.0",
            "sphinx>=5.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "digital-twin-t1d=digital_twin_t1d.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
        "examples": ["*.py", "*.ipynb"],
        "data": ["*.csv", "*.h5", "*.parquet"],
    },
    keywords=[
        "diabetes", "type-1-diabetes", "digital-twin", "machine-learning", 
        "artificial-intelligence", "glucose-prediction", "insulin-optimization",
        "reinforcement-learning", "healthcare", "medical-ai", "cgm", "glucose-monitoring"
    ],
    zip_safe=False,
) 