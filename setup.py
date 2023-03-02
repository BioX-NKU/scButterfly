from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="scButterfly",
    version="0.0.1",
    description="Single-cell cross-modality translation via multi-use dual-aligned variational autoencoders",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    keywords="single cell, cross-modality translation, dual-aligned variational autoencoder",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
)