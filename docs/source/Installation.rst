Installation
============

It's prefered to create a new environment for scButterfly::


    conda create scButterfly python==3.9
    conda activate scButterfly


scButterfly is available on PyPI, and could be installed using::

    # CUDA 11.6
    pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu116
    # CUDA 11.3
    pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu113
    # CUDA 10.2
    pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu102
    # CPU only
    pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cpu

You could choose a CUDA version proper for GPU settings. CPU version will provide the same performance as GPU version, but takes more time for training.

All dependencies will be automatically installed along with scButterfly.

Installation via Github is also provided::


    git clone https://github.com/Biox-NKU/scButterfly
    cd scButterfly

    # CUDA 11.6
    pip install scButterfly.whl --extra-index-url https://download.pytorch.org/whl/cu116

    # CUDA 11.3
    pip install scButterfly.whl --extra-index-url https://download.pytorch.org/whl/cu113

    # CUDA 10.2
    pip install scButterfly.whl --extra-index-url https://download.pytorch.org/whl/cu102

    # CPU only
    pip install scButterfly.whl --extra-index-url https://download.pytorch.org/whl/cpu
