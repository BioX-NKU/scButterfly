import getopt
import sys
import gc
import os
sys.path.append('/home/atac2rna/program/atac2rna/Model/butterfly/Butterfly/')
from data_processing import RNA_data_preprocessing, ATAC_data_preprocessing
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

opts, args = getopt.gnu_getopt(sys.argv[1:], 't:d:f:n', ['model_type=', 'data=', 'file_path=', 'number='])
model_type = opts[0][1]
data = opts[1][1]
file_path = opts[2][1]
number = opts[3][1]

    
if model_type == 'totalVI_amplification':
    from scvi_colab import install
    install()
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scvi
    import sys
    from calculate_cluster import *
    import scipy.sparse as sp
    from split_datasets import *

    if data == 'bmmc_adt':

        sc_data = sc.read_h5ad('/home/atac2rna/data/atac2rna/data/openproblems_neurips2021/GSE194122_openproblems_neurips2021_cite_BMMC_count.h5ad')
        adata_gene = sc_data[:, 0:13953]
        adata_protein = sc_data[:, 13953:]

    if data == 'seurat':

        adata_protein = sc.read_h5ad('/data/cabins/chenshengquan/CITEseq/seurat_ADT_cnt.h5ad')
        adata_gene = sc.read_h5ad('/data/cabins/chenshengquan/CITEseq/seurat_RNA_cnt.h5ad')
     
    sc.pp.highly_variable_genes(adata_gene, batch_key="batch", flavor="seurat_v3", n_top_genes=4000, subset=True)
    
    sys.path.append('/home/atac2rna/program/atac2rna/Model/22_10_10')
    id_list = five_hold_split_dataset(adata_gene, adata_protein, seed=19191)
    train_id, validation_id, test_id = id_list[int(number) - 1]
    
    adata_gene_train = adata_gene[train_id].copy()
    adata_protein_train = adata_protein[train_id].copy()
    
    adata_gene_train.obsm['protein_expression'] = adata_protein_train.to_df()
    
    scvi.model.TOTALVI.setup_anndata(adata_gene_train, batch_key="batch", protein_expression_obsm_key="protein_expression")
    
    model = scvi.model.TOTALVI(adata_gene_train, latent_distribution="normal", n_layers_decoder=2)
    
    model.train()
    
    adata_gene_train.obsm["X_totalVI"] = model.get_latent_representation()
    
    leiden_adata = ad.AnnData(adata_gene_train.obsm["X_totalVI"])
    sc.pp.neighbors(leiden_adata)
    sc.tl.leiden(leiden_adata, resolution=3)


if data == 'bmmc_adt':
    
    ATAC_data = sc.read_h5ad('/home/atac2rna/data/atac2rna/data/openproblems_neurips2021/bmmc_ADT_CLRed.h5ad')
    RNA_data = sc.read_h5ad('/home/atac2rna/data/atac2rna/data/openproblems_neurips2021/bmmc_RNA_cnt.h5ad')
    ATAC_data.X = csr_matrix(ATAC_data.X)
    RNA_data.X = csr_matrix(RNA_data.X)
    
elif data == 'seurat':
    
    ATAC_data = sc.read_h5ad('/home/atac2rna/data/atac2rna/data/CITE_seq/seurat_ADT_CLRed.h5ad')
    RNA_data = sc.read_h5ad('/home/atac2rna/data/atac2rna/data/CITE_seq/seurat_RNA_cnt.h5ad')
    ATAC_data.X = csr_matrix(ATAC_data.X)

############################################################
# Part 1 data processing
RNA_data = RNA_data_preprocessing(
    RNA_data,
    normalize_total=True,
    log1p=True,
    use_hvg=True,
    n_top_genes=3000,
    save_data=False,
    file_path=file_path,
    logging_path=file_path
    )

############################################################
# Part 2 split datasets
from split_datasets import *
id_list = five_hold_split_dataset(RNA_data, ATAC_data, seed=19191)
train_id, validation_id, test_id = id_list[int(number) - 1]
train_id_r = train_id.copy()
train_id_a = train_id.copy()
validation_id_r = validation_id.copy()
validation_id_a = validation_id.copy()
test_id_r = test_id.copy()
test_id_a = test_id.copy()

############################################################
# Part 2.5 amplification
import random
if model_type == 'totalVI_amplification':
    copy_count = 3
    random.seed(19193)
    cell_type = leiden_adata.obs.leiden
    for i in range(len(cell_type.cat.categories)):
        cell_type_name = cell_type.cat.categories[i]
        idx_temp = list(cell_type[cell_type == cell_type_name].index.astype(int))
        for j in range(copy_count - 1):
            random.shuffle(idx_temp)
            for each in idx_temp:
                train_id_r.append(train_id[each])
            random.shuffle(idx_temp)
            for each in idx_temp:
                train_id_a.append(train_id[each])

if model_type == 'celltype_amplification':
    copy_count = 3
    random.seed(19193)
    ATAC_data.obs.index = [str(i) for i in range(len(ATAC_data.obs.index))]
    cell_type = ATAC_data.obs.cell_type.iloc[train_id]
    for i in range(len(cell_type.cat.categories)):
        cell_type_name = cell_type.cat.categories[i]
        idx_temp = list(cell_type[cell_type == cell_type_name].index.astype(int))
        for j in range(copy_count - 1):
            random.shuffle(idx_temp)
            train_id_r.extend(idx_temp)
            random.shuffle(idx_temp)
            train_id_a.extend(idx_temp)

from train_model_cite import Model
import torch
import torch.nn as nn

RNA_input_dim = len([i for i in RNA_data.var['highly_variable'] if i])
ATAC_input_dim = ATAC_data.X.shape[1]

R_kl_div = 1 / RNA_input_dim * 20
A_kl_div = R_kl_div
kl_div = R_kl_div + A_kl_div

############################################################
# Part 4 construct model
model = Model(
    R_encoder_nlayer = 2, 
    A_encoder_nlayer = 2,
    R_decoder_nlayer = 2, 
    A_decoder_nlayer = 2,
    R_encoder_dim_list = [RNA_input_dim, 256, 128],
    A_encoder_dim_list = [ATAC_input_dim, 128, 128],
    R_decoder_dim_list = [128, 256, RNA_input_dim],
    A_decoder_dim_list = [128, 128, ATAC_input_dim],
    R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    A_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
    translator_embed_dim = 128, 
    translator_input_dim_r = 128,
    translator_input_dim_a = 128,
    translator_embed_act_list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
    discriminator_nlayer = 1,
    discriminator_dim_list_R = [128],
    discriminator_dim_list_A = [128],
    discriminator_act_list = [nn.Sigmoid()],
    dropout_rate = 0.1,
    R_noise_rate = 0.5,
    A_noise_rate = 0,
    chrom_list = [],
    logging_path = file_path,
    RNA_data = RNA_data,
    ATAC_data = ATAC_data
)
############################################################
# Part 5 train model

model.train(
    R_encoder_lr = 0.001,
    A_encoder_lr = 0.001,
    R_decoder_lr = 0.001,
    A_decoder_lr = 0.001,
    R_translator_lr = 0.001,
    A_translator_lr = 0.001,
    translator_lr = 0.001,
    discriminator_lr = 0.005,
    R2R_pretrain_epoch = 100,
    A2A_pretrain_epoch = 100,
    lock_encoder_and_decoder = False,
    translator_epoch = 200,
    patience = 50,
    batch_size = 64,
    r_loss = nn.MSELoss(size_average=True),
    a_loss = nn.MSELoss(size_average=True),
    d_loss = nn.BCELoss(size_average=True),
    loss_weight = [1, 2, 1, R_kl_div, A_kl_div, kl_div],
    train_id_r = train_id_r,
    train_id_a = train_id_a,
    validation_id_r = validation_id_r, 
    validation_id_a = validation_id_a, 
    output_path = file_path,
    seed = 19193,
    kl_mean = True,
    R_pretrain_kl_warmup = 50,
    A_pretrain_kl_warmup = 50,
    translation_kl_warmup = 50,
    load_model = None,
    logging_path = file_path
)

############################################################
# Part 6 test model 
model.test(
    test_id_r = test_id_r,
    test_id_a = test_id_a, 
    model_path = file_path,
    load_model = True,
    output_path = file_path,
    test_evaluate = True,
    test_cluster = True,
    test_figure = True,
    output_data = False,
    return_predict = False
)