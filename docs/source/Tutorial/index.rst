
Tutorial
========

Translation between scRNA-seq and scATAC-seq paired data
--------------------------------------------------------

scButterfly could make translation between scRNA-seq and scATAC-seq paired data 
using integrated scButterfly model in ``scButterfly.butterfly``.

* `Basic scButterfly usage <RNA_ATAC_paired_prediction/RNA_ATAC_paired_basic.ipynb>`_
* `scButterfly with data amplification using cell type labels <RNA_ATAC_paired_prediction/RNA_ATAC_paired_celltype_amp.ipynb>`_
* `scButterfly with data amplification using MultiVI integrated labels <RNA_ATAC_paired_prediction/RNA_ATAC_paired_MultiVI_amp.ipynb>`_

Extension usages of scButterfly framework
---------------------------------------

scButterfly provide a series of extension usage with preprocessing in ``scButterfly.data_processing`` and 
model in ``scButterfly.train_model`` (scRNA-seq ~ scATAC-seq), ``scButterfly.train_model_cite`` (scRNA-seq ~ scADT-seq) 
or ``scButterfly.train_model_perturb`` (single-cell perturb data).

Translation between scRNA-seq and scATAC-seq unpaired data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Single omics unpaired data prediction using scButterfly framework <RNA_ATAC_unpaired_prediction/RNA_ATAC_unpaired_basic.ipynb>`_


Translation between scRNA-seq and scADT-seq paired data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Basic scButterfly usage <RNA_ADT_paired_prediction/RNA_ADT_paired_basic.ipynb>`_
* `scButterfly with data amplification using cell type labels <RNA_ADT_paired_prediction/RNA_ADT_paired_celltype_amp.ipynb>`_
* `scButterfly with data amplification using TotalVI integrated labels <RNA_ADT_paired_prediction/RNA_ADT_paired_TotalVI_amp.ipynb>`_


Translation between single cell perturb data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Perturb prediction using scButterfly framework <Perturb_prediction/Perturb_unpaired_basic.ipynb>`_


Examples
--------

.. toctree::
    :maxdepth: 2
    :hidden:

    RNA_ATAC_paired_prediction/RNA_ATAC_paired_basic
    RNA_ATAC_paired_prediction/RNA_ATAC_paired_celltype_amp
    RNA_ATAC_paired_prediction/RNA_ATAC_paired_MultiVI_amp

    RNA_ATAC_unpaired_prediction/RNA_ATAC_unpaired_basic

    RNA_ADT_paired_prediction/RNA_ADT_paired_basic
    RNA_ADT_paired_prediction/RNA_ADT_paired_celltype_amp
    RNA_ADT_paired_prediction/RNA_ADT_paired_TotalVI_amp

    Perturb_prediction/Perturb_unpaired_basic
