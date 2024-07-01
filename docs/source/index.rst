|PyPI| |Docs| |PyPIDownloads|


.. |PyPI| image:: https://img.shields.io/pypi/v/scbutterfly
   :target: https://pypi.org/project/scbutterfly/
.. |Docs| image:: https://readthedocs.org/projects/scbutterfly/badge/?version=latest
   :target: https://scbutterfly.readthedocs.io
.. |PyPIDownloads| image:: https://pepy.tech/badge/scbutterfly
   :target: https://pepy.tech/project/scbutterfly



scButterfly: a versatile single-cell cross-modality translation method via dual-aligned variational autoencoders
==================================================================================================================

Recent advancements for simultaneously profiling multi-omic modalities within individual cells have enabled the interrogation of cellular heterogeneity and molecular hierarchy. However, technical limitations lead to highly noisy multi-modal data and substantial costs. Although computational methods have been proposed to translate single-cell data across modalities, b broad applications of the methods still remain impeded b by formidable challenges. Here, we proposed scButterfly, a versatile single-cell cross-modality translation method based on dual-aligned variational autoencoders and innovative data augmentation schemes. With comprehensive experiments on multiple datasets, we provide compelling evidence of scButterfly's superiority over baseline methods in preserving cellular heterogeneity while translating d datasets o of various contexts and i in revealing cell type-specific biological insights.Besides, we d demonstrate the extensive applications of scButterfly for integrative multi-omics analysis of single-modality data, data enhancement of poor-quality single-cell multi-omics, and automatic cell type annotation of scATAC-seq data. Additionally, scButterfly can be generalized to unpaired data training and perturbation-response analysis via our data augmentation and optimal transport strategies. Moreover, scButterfly exhibits the capability i in consecutive translation from epigenome to transcriptome to proteome and has the potential to decipher novel biomarkers.

.. toctree::
   :maxdepth: 2
   :hidden:
   
   API/index
   Tutorial/index
   Installation
   release/index


News
----

.. include:: news.rst
   :start-line: 2
   :end-line: 22  
