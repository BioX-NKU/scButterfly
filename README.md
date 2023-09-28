[![PyPI](https://img.shields.io/pypi/v/scbutterfly)](https://pypi.org/project/scbutterfly)
[![Documentation Status](https://readthedocs.org/projects/scbutterfly/badge/?version=latest)](https://scbutterfly.readthedocs.io/en/latest/?badge=stable)
[![Downloads](https://pepy.tech/badge/scbutterfly)](https://pepy.tech/project/scbutterfly)


# scButterfly: a versatile single-cell cross-modality translation method via dual-aligned variational autoencoders

## Installation

It's prefered to create a new environment for scButterfly

```
conda create -n scButterfly python==3.9
conda activate scButterfly
```

scButterfly is available on PyPI, and could be installed using

```
# CUDA 11.6
pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.3
pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu113

# CUDA 10.2
pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cu102

# CPU only
pip install scButterfly --extra-index-url https://download.pytorch.org/whl/cpu
```

Installation via Github is also provided

```
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
```

This process will take approximately 5 to 10 minutes, depending on the user's computer device and internet connectivition.

## Quick Start

Illustrating with the translation between  scRNA-seq and scATAC-seq data as an example, scButterfly could be easily used following 3 steps: data preprocessing, model training, predicting and evaluating. More details could be find in [scButterfly documents](http://scbutterfly.readthedocs.io/).

Generate a scButterfly model first with following process:

```python
from scButterfly.butterfly import Butterfly
butterfly = Butterfly()
```

### 1. Data preprocessing

* Before data preprocessing, you should load scRNA-seq and scATAC-seq data via `butterfly.load_data`:
  
  ```python
  butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
  ```
  
  | Parameters    | Description                                                                                |
  | ------------- | ------------------------------------------------------------------------------------------ |
  | RNA_data      | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes. |
  | ATAC_data     | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks. |
  | train_id      | A list of cell IDs for training.                                                           |
  | test_id       | A list of cell IDs for testing.                                                            |
  | validation_id | An optional list of cell IDs for validation, if setted None, butterfly will use a default setting of 20% cells in train_id. |
  
  Anndata object is a Python object/container designed to store single-cell data in Python packege [**anndata**](https://anndata.readthedocs.io/en/latest/) which is seamlessly integrated with [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis.

* For data preprocessing, you could use `butterfly.data_preprocessing`:
  
  ```python
  butterfly.data_preprocessing()
  ```
  
  You could save processed data or output process logging to a file using following parameters.
  
  | Parameters   | Description                                                                                  |
  | ------------ | -------------------------------------------------------------------------------------------- |
  | save_data    | optional, choose save the processed data or not, default False.                              |
  | file_path    | optional, the path for saving processed data, only used if `save_data` is True, default None.  |
  | logging_path | optional, the path for output process logging, if not save, set it None, default None.       |

  scButterfly also support to refine this process using other parameters (more details on [scButterfly documents](http://scbutterfly.readthedocs.io/)), however, we strongly recommend the default settings to keep the best result for model.
  
### 2. Model training

* Before model training, you could choose to use data augmentation strategy or not. If using data augmentation, scButterfly will generate synthetic samgles with the use of cell-type labels(if `cell_type` in `adata.obs`) or cluster labels get with Leiden algorithm and [**MultiVI**](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html), a single-cell multi-omics data joint analysis method in Python packages [**scvi-tools**](https://docs.scvi-tools.org/en/stable/).

  scButterfly provide data augmentation API:
  
  ```python
  butterfly.augmentation(aug_type)
  ```

  You could choose parameter `aug_type` from `cell_type_augmentation` or `MultiVI_augmentation`, this will cause more training time used, but promise better result for predicting. 
  
  * If you choose `cell_type_augmentation`, scButterfly-T (Type) will try to find `cell_type` in `adata.obs`. If failed, it will automaticly transfer to `MultiVI_augmentation`.
  * If you choose `MultiVI_augmentation`, scButterfly-C (Cluster) will train a MultiVI model first.
  * If you just want to using original data for scButterfly-B (Basic) training, set `aug_type = None`.
  
* You could construct a scButterfly model as following:
  
  ```python
  butterfly.construct_model(chrom_list)
  ```
  
  scButterfly need a list of peaks count for each chromosome, remember to sort peaks with chromosomes.
  
  | Parameters   | Description                                                                                    |
  | ------------ | ---------------------------------------------------------------------------------------------- |
  | chrom_list   | a list of peaks count for each chromosome, remember to sort peaks with chromosomes.            |
  | logging_path | optional, the path for output model structure logging, if not save, set it None, default None. |
  
* scButterfly model could be easily trained as following:
  
  ```python
  butterfly.train_model()
  ```

  | Parameters   | Description                                                                             |
  | ------------ | --------------------------------------------------------------------------------------- |
  | output_path  | optional, path for model check point, if None, using './model' as path, default None.   |
  | load_model   | optional, the path for load pretrained model, if not load, set it None, default None.   |
  | logging_path | optional, the path for output training logging, if not save, set it None, default None. |
  
  scButterfly also support to refine the model structure and training process using other parameters for `butterfly.construct_model()` and `butterfly.train_model()` (more details on [scButterfly documents](http://scbutterfly.readthedocs.io/)).
  
### 3. Predicting and evaluating

* scButterfly provide a predicting API, you could get predicted profiles as follow:
  
  ```python
  A2R_predict, R2A_predict = butterly.test_model()
  ```
  
  A series of evaluating method also be integrated in this function, you could get these evaluation using parameters:
  
  | Parameters    | Description                                                                                 |
  | ------------- | ------------------------------------------------------------------------------------------- |
  | output_path   | optional, path for model evaluating output, if None, using './model' as path, default None. |
  | load_model    | optional, the path for load pretrained model, if not load, set it None, default False.      |
  | model_path    | optional, the path for pretrained model, only used if `load_model` is True, default None.   |
  | test_cluster  | optional, test the correlation evaluation or not, including **AMI**, **ARI**, **HOM**, **NMI**, default False.|
  | test_figure   | optional, draw the **tSNE** visualization for prediction or not, default False.             |
  | output_data   | optional, output the prediction to file or not, if True, output the prediction to `output_path/A2R_predict.h5ad` and `output_path/R2A_predict.h5ad`, default False.                                          |

## Demo, document, tutorial and source code

### We provide demos of basic scButterfly model and two variants (scButterfly-C and scButterfly-T) illustrating with CL datasets in [scButterfly-B usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-B.html), [scButterfly-C usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-C.html), and [scButterfly-T usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-T.html), with data presented in [Google drive](https://drive.google.com/drive/folders/1CAZp11EF1t6szAc__m2ceNbMd5KlbqPJ). scButterfly-B, scButterfly-C and scButterfly-T repectively take about 12, 24, 18 minutes for the whole process (containing pre-processing, data augmentation, model training and evaluating) on desktop computer with NVIDIA RTX A6000 GPU.

### We also provide richer tutorials and documents for scButterfly in [scButterfly documents](http://scbutterfly.readthedocs.io/), including more details of provided APIs for customing data preprocessing, model structure and training strategy. The source code of experiments for scButterfly is available at [source code](https://github.com/BioX-NKU/scButterfly_source), including more detailed source code for scButterfly.
