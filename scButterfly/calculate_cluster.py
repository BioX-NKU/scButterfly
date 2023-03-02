import scanpy as sc
from sklearn import metrics
from scButterfly.logger import *


def calculate_cluster_index(adata):
    """
    Evaluation of cluster index of prediction. 
    Cluster method using Leiden with a default resolution.
    5 different metrics provided: ARI - Adjusted Rand Index, NMI - Normalized Mutual Information, AMI - Adjusted Mutual Information, HOM - Homogeneity, COM - Completeness.
    
    Parameters
    ----------
    adata: Anndata
        Anndata need to cumpute index, there need ground truth labels "cell_type" in adata.obs.
   
    Returns
    ----------
    ARI: float
        Adjusted Rand Index.
    AMI: float
        Normalized Mutual Information.
    NMI: float
        Adjusted Mutual Information.
    HOM: float
        Homogeneity.
    COM: float
        Completeness.
    """
    my_logger = create_logger(name='measure performance', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if not "neighbors" in adata.uns.keys():
        my_logger.warning('No "neighbors" in .uns, create "neighbors" first.')
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
    
    sc.tl.leiden(adata)
    #sc.tl.louvain(adata)
    
    ARI = metrics.adjusted_rand_score(adata.obs['cell_type'], adata.obs['leiden'])
    AMI = metrics.adjusted_mutual_info_score(adata.obs['cell_type'], adata.obs['leiden'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['cell_type'], adata.obs['leiden'])
    HOM = metrics.homogeneity_score(adata.obs['cell_type'], adata.obs['leiden'])
    COM = metrics.completeness_score(adata.obs['cell_type'], adata.obs['leiden'])
    
    return ARI, AMI, NMI, HOM, COM