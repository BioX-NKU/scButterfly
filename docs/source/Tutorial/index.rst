
Tutorial
========

Translation between scRNA-seq and scATAC-seq paired data
--------------------------------------------------------

scButterfly could make translation between scRNA-seq and scATAC-seq paired data using scButterfly model in ``scButterfly.butterfly``.

* `scButterfly-B usage <RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-B.ipynb>`_
* `scButterfly-T using augmentation with cell-type labels <RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-T.ipynb>`_
* `scButterfly-C using augmentation with MultiVI integrated labels <RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-C.ipynb>`_

Extension usages of scButterfly framework
---------------------------------------

scButterfly provide a series of extension usage with preprocessing in ``scButterfly.data_processing`` and 
model in ``scButterfly.train_model`` (scRNA-seq ~ scATAC-seq), ``scButterfly.train_model_cite`` (scRNA-seq ~ ADT data) 
or ``scButterfly.train_model_perturb`` (single-cell perturbation responses data).

Translation between scRNA-seq and scATAC-seq unpaired data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Unpaired data translation using scButterfly-T <RNA_ATAC_unpaired_prediction/RNA_ATAC_unpaired_scButterfly-T.ipynb>`_


Translation between scRNA-seq and scADT-seq paired data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `scButterfly-B usage <RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-B.ipynb>`_
* `scButterfly-T using augmentation with cell-type labels <RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-T.ipynb>`_
* `scButterfly-C using augmentation with totalVI integrated labels <RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-C.ipynb>`_


Translation between single-cell perturbation responses data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* `Perturbation responses prediction using scButterfly-T <Perturb_prediction/Perturb_unpaired_scButterfly-T.ipynb>`_


Examples
--------

.. toctree::
    :maxdepth: 2
    :hidden:

    RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-B
    RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-T
    RNA_ATAC_paired_prediction/RNA_ATAC_paired_scButterfly-C

    RNA_ATAC_unpaired_prediction/RNA_ATAC_unpaired_scButterfly-T

    RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-B
    RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-T
    RNA_ADT_paired_prediction/RNA_ADT_paired_scButterfly-C

    Perturb_prediction/Perturb_unpaired_scButterfly-T
