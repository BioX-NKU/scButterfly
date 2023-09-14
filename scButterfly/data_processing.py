import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import numpy as np
import random
import episcanpy.api as epi
import scipy
from scipy.sparse import csr_matrix
from scButterfly.logger import *


def TFIDF(count_mat): 
    """
    TF-IDF transformation for matrix.

    Parameters
    ----------
    count_mat
        numpy matrix with cells as rows and peak as columns, cell * peak.

    Returns
    ----------
    tfidf_mat
        matrix after TF-IDF transformation.

    divide_title
        matrix divided in TF-IDF transformation process, would be used in "inverse_TFIDF".

    multiply_title
        matrix multiplied in TF-IDF transformation process, would be used in "inverse_TFIDF".

    """

    count_mat = count_mat.T
    divide_title = np.tile(np.sum(count_mat,axis=0), (count_mat.shape[0],1))
    nfreqs = 1.0 * count_mat / divide_title
    multiply_title = np.tile(np.log(1 + 1.0 * count_mat.shape[1] / np.sum(count_mat,axis=1)).reshape(-1,1), (1,count_mat.shape[1]))
    tfidf_mat = scipy.sparse.csr_matrix(np.multiply(nfreqs, multiply_title)).T
    return tfidf_mat, divide_title, multiply_title


def inverse_TFIDF(TDIDFed_mat, divide_title, multiply_title, max_temp): 
    """
    Inversed TF-IDF transformation for matrix.

    Parameters
    ----------
    TDIDFed_mat: csr_matrix
        matrix after TFIDF transformation with peaks as rows and cells as columns, peak * cell.
        
    divide_title: numpy matrix
        matrix divided in TF-IDF transformation process, could get from "ATAC_data_preprocessing".
        
    multiply_title: numpy matrix
        matrix multiplied in TF-IDF transformation process, could get from "ATAC_data_preprocessing".
        
    max_temp: float
        max scale factor divided in ATAC preprocessing, could get from "ATAC_data_preprocessing".
        
    Returns
    ----------
    count_mat: csr_matrix
        recovered count matrix from matrix after TFIDF transformation.
    """
    
    count_mat = TDIDFed_mat.T
    count_mat = count_mat * max_temp
    nfreqs = np.divide(count_mat, multiply_title)
    count_mat = np.multiply(nfreqs, divide_title).T
    return count_mat


def RNA_data_preprocessing(
    RNA_data,
    normalize_total = True,
    log1p = True,
    use_hvg = True,
    n_top_genes = 3000,
    save_data = False,
    file_path = None,
    logging_path = None
):
    """
    Preprocessing for RNA data, we choose normalization, log transformation and highly variable genes, using scanpy.
    
    Parameters
    ----------
    RNA_data: Anndata
        RNA anndata for processing.
        
    normalize_total: bool
        choose use normalization or not, default True.
        
    log1p: bool
        choose use log transformation or not, default True.
        
    use_hvg: bool
        choose use highly variable genes or not, default True.
        
    n_top_genes: int
        the count of highly variable genes, if not use highly variable, set use_hvg = False and n_top_genes = None, default 3000.
        
    save_data: bool
        choose save the processed data or not, default False.
        
    file_path: str
        the path for saving processed data, only used if save_data is True, default None.
   
    logging_path: str
        the path for output process logging, if not save, set it None, default None.

    Returns
    ---------
    RNA_data_processed: Anndata
        RNA data with normalization, log transformation and highly variable genes selection preprocessed.
    """
    RNA_data_processed = RNA_data.copy()
    
    my_logger = create_logger(name='RNA preprocessing', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if not logging_path is None:
        file_handle=open(logging_path + '/Parameters_Record.txt',mode='w')
        file_handle.writelines([
            '------------------------------\n',
            'RNA Preprocessing\n',
            'normalize_total: '+str(normalize_total)+'\n', 
            'log1p: '+str(log1p)+'\n',
            'use_hvg: '+str(use_hvg)+'\n', 
            'n_top_genes: '+str(n_top_genes)+'\n'
        ])
        file_handle.close()
    
    RNA_data_processed.var_names_make_unique()
    
    if not normalize_total or not log1p or not use_hvg:
        my_logger.warning('prefered to process data with default settings to keep the best result.')
    
    if normalize_total:
        my_logger.info('normalize size factor.')
        sc.pp.normalize_total(RNA_data_processed)
        
    if log1p:
        my_logger.info('log transform RNA data.')
        sc.pp.log1p(RNA_data_processed)
    
    if use_hvg:
        my_logger.info('choose top '+str(n_top_genes)+' genes for following training.')
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var['highly_variable']]
    
    if save_data:
        my_logger.warning('writing processed RNA data to target file.')
        if use_hvg:
            RNA_data_processed.write_h5ad(file_path + '/normalize_' + str(normalize_total) + '_log1p_' + str(log1p) + '_hvg_' + str(use_hvg) + '_' + str(n_top_genes) + '_RNA_processed_data.h5ad')
        else:
            RNA_data_processed.write_h5ad(file_path + '/normalize_' + str(normalize_total) + '_log1p_' + str(log1p) + '_hvg_' + str(use_hvg) + '_RNA_processed_data.h5ad')
            
    return RNA_data_processed


def ATAC_data_preprocessing(
    ATAC_data,
    binary_data = True,
    filter_features = True,
    fpeaks = 0.005,
    tfidf = True,
    normalize = True,
    save_data = False,
    file_path = None,
    logging_path = None
):
    """
    Preprocessing for ATAC data, we choose binarize, peaks filtering, TF-IDF transformation and scale transformation, using scanpy.
    
    Parameters
    ----------
    ATAC_data: Anndata
        ATAC anndata for processing.
        
    binary_data: bool
        choose binarized ATAC data or not, default True.
        
    filter_features: bool
        choose use peaks filtering or not, default True.
        
    fpeaks: float
        the fpeaks for peaks filtering, is don't filter peaks set it None, default 0.005.
        
    tfidf: bool
        choose using TF-IDF transform or not, default True.
    
    normalize: bool
        choose set data to [0, 1] or not, default True.
        
    save_data: bool
        choose save the processed data or not, default False.
        
    file_path: str
        the path for saving processed data, only used if save_data is True, default None.
   
    logging_path: str
        the path for output process logging, if not save, set it None, default None.

    Returns
    ---------
    ATAC_data_processed: Anndata 
        ATAC data with binarization, peaks filtering, TF-IDF transformation and scale transformation preprocessed.

    divide_title: numpy matrix
        matrix divided in TF-IDF transformation process, would be used in "inverse_TFIDF".

    multiply_title: numpy matrix
        matrix multiplied in TF-IDF transformation process, would be used in "inverse_TFIDF".
        
    max_temp: float
        max scale factor divided in process, would be used in "inverse_TFIDF".
        
    """
    ATAC_data_processed = ATAC_data.copy()
    
    my_logger = create_logger(name='ATAC preprocessing', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    if not logging_path is None:
        file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
        file_handle.writelines([
            '------------------------------\n'
            'ATAC Preprocessing\n'
            'binary_data: '+str(binary_data)+'\n', 
            'filter_features: '+str(filter_features)+'\n',
            'fpeaks: '+str(fpeaks)+'\n', 
            'tfidf: '+str(tfidf)+'\n',
            'normalize: '+str(normalize)+'\n'
        ])
        file_handle.close()
    
    if not binary_data or not filter_features or not tfidf or not normalize:
        my_logger.warning('prefered to process data with default settings to keep the best result.')
    
    if binary_data:
        my_logger.info('binarizing data.')
        epi.pp.binarize(ATAC_data_processed)
        
    if filter_features:
        my_logger.info('filter out peaks appear lower than ' + str(fpeaks * 100) + '% cells.')
        epi.pp.filter_features(ATAC_data_processed, min_cells=np.ceil(fpeaks*ATAC_data.shape[0]))

    if tfidf:
        my_logger.info('TF-IDF transformation.')
        count_mat = ATAC_data_processed.X.copy()
        ATAC_data_processed.X, divide_title, multiply_title = TFIDF(count_mat)
    
    if normalize:
        my_logger.info('normalizing data.')
        max_temp = np.max(ATAC_data_processed.X)
        ATAC_data_processed.X = ATAC_data_processed.X / max_temp
    
    if save_data:
        my_logger.warning('writing processed ATAC data to target file.')
        if filter_features:
            ATAC_data.write_h5ad(file_path + '/binarize_' + str(binary_data) + '_filter_' + str(filter_features) + '_fpeaks_' + str(fpeaks)  + '_tfidf_' + str(tfidf) + '_normalize_' + str(normalize) + '_ATAC_processed_data.h5ad')
        else:
            ATAC_data.write_h5ad(file_path + '/binarize_' + str(binary_data) + '_filter_' + str(filter_features) + '_tfidf_' + str(tfidf) + '_normalize_' + str(normalize) + '_ATAC_processed_data.h5ad')

    return ATAC_data_processed, divide_title, multiply_title, max_temp