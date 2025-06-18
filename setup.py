from setuptools import setup, find_packages

setup(
    name="pyrifreg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "statsmodels>=0.13.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for Recentered Influence Function (RIF) regression",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vyasenov/pyrifreg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 