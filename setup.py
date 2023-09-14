from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="scButterfly",
    version="0.0.5",
    description="Single-cell cross-modality translation via multi-use dual-aligned variational autoencoders",
    long_description="Single-cell cross-modality translation via multi-use dual-aligned variational autoencoders",
    license="MIT Licence",
    url="https://github.com/BioX-NKU/scButterfly",
    author="BioX-NKU",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    keywords="single cell, cross-modality translation, dual-aligned variational autoencoder",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        'scanpy>=1.9.1',
        'torch==1.12.1',
        'torchvision==0.13.1',
        'torchaudio==0.12.1',
        'scikit-learn>=1.1.3',
        'scvi-tools==0.19.0',
        'scvi-colab',
        'scipy>=1.9.3',
        'episcanpy==0.3.2',
        'seaborn>=0.11.2',
        'matplotlib>=3.6.2',
        'pot==0.9.0',
        'leidenalg',
        'pybedtools',
        'adjusttext',
        'jupyter'
    ]
)