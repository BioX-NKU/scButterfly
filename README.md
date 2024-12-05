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

scButterfly is available on PyPI, and can be installed using

```
pip install scButterfly
```

Installation via Github is also possible

```
git clone https://github.com/Biox-NKU/scButterfly
cd scButterfly
pip install scButterfly-0.0.9-py3-none-any.whl
```

This process will take approximately 5 to 10 minutes, depending on the user's computer device and internet connection.

## Quick Start

Illustrating with the translation between  scRNA-seq and scATAC-seq data as an example, scButterfly can be easily used following these 3 steps: data preprocessing, model training, and predicting and evaluating. More details can be found in the [scButterfly documentation](http://scbutterfly.readthedocs.io/).

First, generate a scButterfly model with the following code:

```python
from scButterfly.butterfly import Butterfly
butterfly = Butterfly()
```

### 1. Data preprocessing

* Before data preprocessing, you should load the **raw count matrix** of scRNA-seq and scATAC-seq data using `butterfly.load_data`:
  
  ```python
  butterfly.load_data(RNA_data, ATAC_data, train_id, test_id, validation_id)
  ```
  
  | Parameters    | Description                                                                                |
  | ------------- | ------------------------------------------------------------------------------------------ |
  | RNA_data      | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to genes. |
  | ATAC_data     | AnnData object of shape `n_obs` × `n_vars`. Rows correspond to cells and columns to peaks. |
  | train_id      | A list of cell IDs for training.                                                           |
  | test_id       | A list of cell IDs for testing.                                                            |
  | validation_id | An optional list of cell IDs for validation, if set to None, butterfly will use a default setting of 20% of the cells in train_id. |
  
  The Anndata object is a Python object/container designed to store single-cell data, available in the Python package [**anndata**](https://anndata.readthedocs.io/en/latest/) which is seamlessly integrated with [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis.

* For data preprocessing, you can use `butterfly.data_preprocessing`:
  
  ```python
  butterfly.data_preprocessing()
  ```
  
  You can save processed data or output process logging to a file using the following parameters:
  
  | Parameters   | Description                                                                                  |
  | ------------ | -------------------------------------------------------------------------------------------- |
  | save_data    | optional, whether to save the preprocessed data or not, default False.                              |
  | file_path    | optional, the path for saving preprocessed data, only used if `save_data` is True, default None.  |
  | logging_path | optional, the path to output process logs. Default None.       |

  scButterfly also supports refining this process using other parameters (more details in the [scButterfly documentation](http://scbutterfly.readthedocs.io/)). However, we strongly recommend the default settings to give the best results.
  
### 2. Model training

* Before model training, you can choose to use a data augmentation strategy. If using data augmentation, scButterfly will generate synthetic samples with the use of cell-type labels(if `cell_type` in `adata.obs`) or cluster labels obtained using the Leiden algorithm and [**MultiVI**](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/MultiVI_tutorial.html), a single-cell multi-omics data joint analysis method from the Python package collection [**scvi-tools**](https://docs.scvi-tools.org/en/stable/).

  scButterfly provides a data augmentation API:
  
  ```python
  butterfly.augmentation(aug_type)
  ```

  You can set the parameter `aug_type` to `cell_type_augmentation` or `MultiVI_augmentation`. Both will increase training time, but improve prediction results. 
  
  * If you choose `cell_type_augmentation`, scButterfly-T (Type) will try to find `cell_type` in `adata.obs`. If this fails, it will automatically switch to `MultiVI_augmentation`.
  * If you choose `MultiVI_augmentation`, scButterfly-C (Cluster) will train a MultiVI model first.
  * If you just want to use the original data for scButterfly-B (Basic) training, set `aug_type = None`.
  
* You can construct a scButterfly model like so:
  
  ```python
  butterfly.construct_model(chrom_list)
  ```
  
  scButterfly needs a list of peak counts for each chromosome. Remember to sort peaks by chromosome.
  
  | Parameters   | Description                                                                                    |
  | ------------ | ---------------------------------------------------------------------------------------------- |
  | chrom_list   | a list of peak counts for each chromosome, remember to sort peaks by chromosome.            |
  | logging_path | optional, the path for output model structure logging, if not save, set it None, default None. |
  
* An scButterfly model can be easily trained using the following code:
  
  ```python
  butterfly.train_model()
  ```

  | Parameters   | Description                                                                             |
  | ------------ | --------------------------------------------------------------------------------------- |
  | output_path  | optional, path for saving model check point, if None, use './model' as path, default None.   |
  | load_model   | optional, path for loading a pretrained model, default None.   |
  | logging_path | optional, the path for output training logging, default None. |
  
  scButterfly also supports refining the model structure and training process using other parameters for `butterfly.construct_model()` and `butterfly.train_model()` (more details in the [scButterfly documentation](http://scbutterfly.readthedocs.io/)).
  
### 3. Predicting and evaluating

* scButterfly provide a prediction API, you can get predicted profiles as follows:
  
  ```python
  A2R_predict, R2A_predict = butterly.test_model()
  ```
  
  A series of evaluating methods are also integrated in this function, you can get these by setting evaluation parameters:
  
  | Parameters    | Description                                                                                 |
  | ------------- | ------------------------------------------------------------------------------------------- |
  | output_path   | optional, path for model evaluation output, if None, uses './model' as path, default None. |
  | load_model    | optional, path for loading a pretrained model, if not load, set it None, default False.      |
  | model_path    | optional, the path for pretrained model, only used if `load_model` is True, default None.   |
  | test_cluster  | optional, whether to evaluate correlations, including **AMI**, **ARI**, **HOM**, **NMI**, default False.|
  | test_figure   | optional, whether to draw the **tSNE** visualization for predictions, default False.             |
  | output_data   | optional, output the predictions to file or not, if True, output the prediction to `output_path/A2R_predict.h5ad` and `output_path/R2A_predict.h5ad`, default False.                                          |

## Demo, document, tutorial and source code

### We provide demos of basic scButterfly model and two variants (scButterfly-C and scButterfly-T) illustrating with CL datasets in [scButterfly-B usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-B.html), [scButterfly-C usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-C.html), and [scButterfly-T usage](https://scbutterfly.readthedocs.io/en/latest/Tutorial/RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-T.html), with data presented in [Google drive](https://drive.google.com/drive/folders/1CAZp11EF1t6szAc__m2ceNbMd5KlbqPJ). scButterfly-B, scButterfly-C and scButterfly-T repectively take about 12, 24, 18 minutes for the whole process (containing pre-processing, data augmentation, model training and evaluating) on desktop computer with NVIDIA RTX A6000 GPU.

### We also provide richer tutorials and documents for scButterfly in [scButterfly documents](http://scbutterfly.readthedocs.io/), including more details of provided APIs for customing data preprocessing, model structure and training strategy. The source code of experiments for scButterfly is available at [source code](https://github.com/BioX-NKU/scButterfly_source), including more detailed source code for scButterfly.
