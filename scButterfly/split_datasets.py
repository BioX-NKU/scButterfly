import scanpy as sc
import anndata as ad
import pandas as pd
import torch
import numpy as np
import random


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    

def idx2adata_multiVI(
    adata,
    train_id,
    validation_id, 
    test_id
):

    train_adata = ad.AnnData(adata.X[train_id, :])
    test_adata = ad.AnnData(adata.X[test_id, :])

    train_adata.var = adata.var
    test_adata.var = adata.var

    train_adata.obs = adata.obs.iloc[train_id, :].copy()
    test_adata.obs = adata.obs.iloc[test_id, :].copy()

    train_adata.obs.reset_index(drop=True, inplace=True)
    test_adata.obs.reset_index(drop=True, inplace=True)

    return train_adata, test_adata


def five_fold_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    
    id_list = []
    
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))
    
    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count: ]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def cell_type_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    cell_type_list = list(RNA_data.obs.cell_type)
    cell_type = list(RNA_data.obs.cell_type.cat.categories)

    random.shuffle(cell_type)
    
    id_list = []
    
    test_count = int(len(cell_type) / 3)
    train_count = int(len(cell_type) / 2)
    validation_count = len(cell_type) - test_count - train_count
    
    for i in range(3):
        test_type = cell_type[: test_count]
        validation_type = cell_type[test_count: test_count + validation_count]
        train_type = cell_type[test_count + validation_count:]
        cell_type.extend(test_type)
        cell_type = cell_type[test_count: ]

        train_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in train_type]
        validation_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in validation_type]
        test_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in test_type]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def bmmc_batch_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    cell_type_list = list(RNA_data.obs_names)
    cell_type = ['1', '2', '3', '4']

    random.shuffle(cell_type)
    
    id_list = []
    
    test_count = 1
    train_count = 2
    validation_count = 1
    
    for i in range(4):
        test_type = cell_type[: test_count]
        validation_type = cell_type[test_count: test_count + validation_count]
        train_type = cell_type[test_count + validation_count:]
        cell_type.extend(test_type)
        cell_type = cell_type[test_count: ]

        train_id = [i for i in range(len(cell_type_list)) if cell_type_list[i].split('-')[-1][1] in train_type]
        validation_id = [i for i in range(len(cell_type_list)) if cell_type_list[i].split('-')[-1][1] in validation_type]
        test_id = [i for i in range(len(cell_type_list)) if cell_type_list[i].split('-')[-1][1] in test_type]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def batch_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    
    if not seed is None:
        setup_seed(seed)
    
    cell_type_list = list(RNA_data.obs.batch)
    cell_type = list(RNA_data.obs.batch.cat.categories)

    random.shuffle(cell_type)
    
    id_list = []
    
    test_count = int(len(cell_type) / 4)
    validation_count = int(len(cell_type) / 4)
    train_count = len(cell_type) - test_count - validation_count
    
    for i in range(4):
        test_type = cell_type[: test_count]
        validation_type = cell_type[test_count: test_count + validation_count]
        train_type = cell_type[test_count + validation_count:]
        cell_type.extend(test_type)
        cell_type = cell_type[test_count: ]

        train_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in train_type]
        validation_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in validation_type]
        test_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] in test_type]

        id_list.append([train_id, validation_id, test_id])
    
    return id_list



def bm_batch_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    if not seed is None:
        setup_seed(seed)
    
    cell_type_list = list(RNA_data.obs.batch)
    cell_type = list(RNA_data.obs.batch.cat.categories)

    random.shuffle(cell_type)
    
    id_list = []
    
    for i in range(2):
        test_type = cell_type[i]
        train_type = cell_type[i - 1]
        
        train_valid_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] == train_type]
        test_id = [i for i in range(len(cell_type_list)) if cell_type_list[i] == test_type]

        random.shuffle(train_valid_id)
        
        train_id = train_valid_id[0: int(0.8 * len(train_valid_id))]
        validation_id = train_valid_id[int(0.8 * len(train_valid_id)):]
        
        id_list.append([train_id, validation_id, test_id])
    
    return id_list


def unpaired_split_dataset_perturb(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):

    import ot
    
    if not seed is None:
        setup_seed(seed)

    cell_type_list = list(RNA_data.obs.cell_type.cat.categories)
    
    id_list = [[[] for j in range(6)] for i in range(len(cell_type_list))]
    
    batch_list = list(RNA_data.obs.cell_type.cat.categories)
    
    random.shuffle(batch_list)
    
    embedding_dict = {}
    for ctype in cell_type_list:
        sc_data_type_R = RNA_data[RNA_data.obs.cell_type == ctype].copy()
        sc_data_type_A = ATAC_data[ATAC_data.obs.cell_type == ctype].copy()
        sc_data_type = sc.AnnData.concatenate(sc_data_type_R, sc_data_type_A)
        sc.pp.pca(sc_data_type)
        z_ctrl = sc_data_type.obsm['X_pca'][sc_data_type.obs.condition == 'control']
        z_stim = sc_data_type.obsm['X_pca'][sc_data_type.obs.condition == 'stimulated']
        M = ot.dist(z_ctrl, z_stim, metric='euclidean')
        G = ot.emd(torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                  torch.ones(z_stim.shape[0]) / z_stim.shape[0],
                  torch.tensor(M), numItermax=100000)
        best_match = G.numpy().argsort()[:, -1:]
        best_match = [each[0] for each in best_match[:, ::-1].tolist()]
        embedding_dict[ctype] = best_match
        print(ctype)
    
    for i in range(len(cell_type_list)):
        
        test_batch_r = batch_list[:1]
        validation_batch_r = batch_list[1:]
        train_batch_r = batch_list[1:]
        test_batch_a = batch_list[:1]
        validation_batch_a = batch_list[1:]
        train_batch_a = batch_list[1:]
        batch_list.extend(test_batch_r)
        batch_list = batch_list[1:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.cell_type[RNA_data.obs.cell_type == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.cell_type[ATAC_data.obs.cell_type == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.cell_type == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.cell_type == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.cell_type == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.cell_type == each].index))
        train_RNA = RNA_data.obs.loc[train_RNA_id, :]
        train_ATAC = ATAC_data.obs.loc[train_ATAC_id, :]
        validation_RNA = RNA_data.obs.loc[validation_RNA_id, :]
        validation_ATAC = ATAC_data.obs.loc[validation_ATAC_id, :]

        for ctype in cell_type_list:
            
            train_r_id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            train_count = int(0.8 * len(train_r_id_temp))
            if train_count == 0:
                continue
            id_list[i][0].extend(train_r_id_temp[0:train_count])
            train_a_id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            train_a_id_temp = np.array(train_a_id_temp)
            id_list[i][1].extend(list(train_a_id_temp[embedding_dict[ctype]])[0:train_count])
            
            validation_r_id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            id_list[i][2].extend(validation_r_id_temp[train_count:])
            validation_a_id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            validation_a_id_temp = np.array(validation_a_id_temp)
            id_list[i][3].extend(list(validation_a_id_temp[embedding_dict[ctype]])[train_count:])
            
    return id_list


def unpaired_split_dataset_perturb_no_reusing(
    RNA_data, 
    ATAC_data,
    sc_data,
    seed = 19193,
):
    import ot
    from scipy.optimize import linear_sum_assignment
    
    if not seed is None:
        setup_seed(seed)

    cell_type_list = list(RNA_data.obs.cell_type.cat.categories)
    
    id_list = [[[] for j in range(6)] for i in range(len(cell_type_list))]
    
    batch_list = list(RNA_data.obs.cell_type.cat.categories)
    
    random.shuffle(batch_list)
    
    embedding_dict_ctr = {}
    embedding_dict_sti = {}
    for ctype in cell_type_list:
        sc_data_type_R = RNA_data[RNA_data.obs.cell_type == ctype].copy()
        sc_data_type_A = ATAC_data[ATAC_data.obs.cell_type == ctype].copy()
        sc_data_type = sc.AnnData.concatenate(sc_data_type_R, sc_data_type_A)
        sc.pp.pca(sc_data_type)
        z_ctrl = sc_data_type.obsm['X_pca'][sc_data_type.obs.condition == 'control']
        z_stim = sc_data_type.obsm['X_pca'][sc_data_type.obs.condition == 'stimulated']
        M = ot.dist(z_ctrl, z_stim, metric='euclidean')
        G = ot.emd(torch.ones(z_ctrl.shape[0]) / z_ctrl.shape[0],
                  torch.ones(z_stim.shape[0]) / z_stim.shape[0],
                  torch.tensor(M), numItermax=100000)
        row_ind,col_ind = linear_sum_assignment(G)
        embedding_dict_ctr[ctype] = row_ind
        embedding_dict_sti[ctype] = col_ind
        print(ctype)
    
    for i in range(len(cell_type_list)):
        
        test_batch_r = batch_list[:1]
        validation_batch_r = batch_list[1:]
        train_batch_r = batch_list[1:]
        test_batch_a = batch_list[:1]
        validation_batch_a = batch_list[1:]
        train_batch_a = batch_list[1:]
        batch_list.extend(test_batch_r)
        batch_list = batch_list[1:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.cell_type[RNA_data.obs.cell_type == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.cell_type[ATAC_data.obs.cell_type == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.cell_type == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.cell_type == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.cell_type == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.cell_type == each].index))
        train_RNA = RNA_data.obs.loc[train_RNA_id, :]
        train_ATAC = ATAC_data.obs.loc[train_ATAC_id, :]
        validation_RNA = RNA_data.obs.loc[validation_RNA_id, :]
        validation_ATAC = ATAC_data.obs.loc[validation_ATAC_id, :]

        for ctype in cell_type_list:
            train_r_id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            train_count = int(0.8 * len(train_r_id_temp))
            if train_count == 0:
                continue
            train_r_id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            train_r_id_temp = np.array(train_r_id_temp)
            id_list[i][0].extend(list(train_r_id_temp[embedding_dict_ctr[ctype]])[0:train_count])
            train_a_id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            train_a_id_temp = np.array(train_a_id_temp)
            id_list[i][1].extend(list(train_a_id_temp[embedding_dict_sti[ctype]])[0:train_count])
            
            validation_r_id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            validation_r_id_temp = np.array(validation_r_id_temp)
            id_list[i][2].extend(list(validation_r_id_temp[embedding_dict_ctr[ctype]])[train_count:])
            validation_a_id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            validation_a_id_temp = np.array(validation_a_id_temp)
            id_list[i][3].extend(list(validation_a_id_temp[embedding_dict_sti[ctype]])[train_count:])
            
    return id_list


def unpaired_split_dataset(
    RNA_data, 
    ATAC_data, 
    seed = 19193
):
    """
    Split datasets into train, validation and test part using different cell types.
    
    Parameters
    ----------
    RNA_data
        full RNA data for spliting.
        
    ATAC_data
        full ATAC data for spliting.
        
    seed
        random seed use to split datasets, if don't give random seed, set it None.
        
    """ 
    
    if not seed is None:
        setup_seed(seed)
    
    RNA_types = list(RNA_data.obs.cell_type.value_counts().index)
    ATAC_types = list(ATAC_data.obs.cell_type.value_counts().index)
    cell_type_list = list(set(RNA_types) & set(ATAC_types))
    cell_type_r_only = list(set(RNA_types) - set(ATAC_types))
    cell_type_a_only = list(set(ATAC_types) - set(RNA_types))
    celltype_distribution_r = RNA_data.obs.cell_type.value_counts(normalize=True)
    celltype_distribution_a = ATAC_data.obs.cell_type.value_counts(normalize=True)
    celltype_distribution = {}
    for each in cell_type_list:
        celltype_distribution[each] = (celltype_distribution_r[each] + celltype_distribution_a[each]) / 2
    sum_celltype = sum(celltype_distribution.values())
    for each in cell_type_list:
        celltype_distribution[each] /= sum_celltype
    
    id_list = [[[] for j in range(6)] for i in range(5)]
    
    RNA_id_list = []
    ATAC_id_list = []
    RNA_id_list_r_only = []
    ATAC_id_list_a_only = []
    for i in range(RNA_data.X.shape[0]):
        if RNA_data.obs.cell_type[i] in cell_type_list:
            RNA_id_list.append(i)
        if RNA_data.obs.cell_type[i] in cell_type_r_only:
            RNA_id_list_r_only.append(i)
    for i in range(ATAC_data.X.shape[0]):
        if ATAC_data.obs.cell_type[i] in cell_type_list:
            ATAC_id_list.append(i)
        if ATAC_data.obs.cell_type[i] in cell_type_a_only:
            ATAC_id_list_a_only.append(i)

    RNA_test_count = int(0.2*len(RNA_id_list))
    RNA_validation_count = int(0.16*len(RNA_id_list))
    ATAC_test_count = int(0.2*len(ATAC_id_list))
    ATAC_validation_count = int(0.16*len(ATAC_id_list))
    RNA_train_count = int(0.64*len(ATAC_id_list))
    ATAC_train_count = int(0.64*len(ATAC_id_list))
    
    max_train_count = (RNA_train_count + ATAC_train_count) / 2
    max_validation_count = (RNA_validation_count + ATAC_validation_count) / 2
    
    random.shuffle(RNA_id_list)
    random.shuffle(ATAC_id_list)
    random.shuffle(RNA_id_list_r_only)
    random.shuffle(ATAC_id_list_a_only)
    
    for i in range(5):
        train_batch_r = RNA_id_list[RNA_test_count+RNA_validation_count:]
        validation_batch_r = RNA_id_list[RNA_test_count:RNA_test_count+RNA_validation_count]
        test_batch_r = RNA_id_list[0:RNA_test_count]
        train_batch_a = ATAC_id_list[ATAC_test_count+ATAC_validation_count:]
        validation_batch_a = ATAC_id_list[ATAC_test_count:ATAC_test_count+ATAC_validation_count]
        test_batch_a = ATAC_id_list[0:ATAC_test_count]
        RNA_id_list.extend(test_batch_r)
        RNA_id_list = RNA_id_list[RNA_test_count:]
        ATAC_id_list.extend(test_batch_a)
        ATAC_id_list = ATAC_id_list[ATAC_test_count:]
        
        id_list[i][4].extend(test_batch_r)

        id_list[i][5].extend(test_batch_a)
        
        train_RNA = RNA_data.obs.iloc[train_batch_r, :]
        train_ATAC = ATAC_data.obs.iloc[train_batch_a, :]
        validation_RNA = RNA_data.obs.iloc[validation_batch_r, :]
        validation_ATAC = ATAC_data.obs.iloc[validation_batch_a, :]
        
        for ctype in cell_type_list:
            train_count = int(max_train_count * celltype_distribution[ctype])
            validation_count = int(max_validation_count * celltype_distribution[ctype])
            
            id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            id_list[i][0].extend(np.random.choice(id_temp, train_count, replace=True))     
            id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][1].extend(np.random.choice(id_temp, train_count, replace=True))
            id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            id_list[i][2].extend(np.random.choice(id_temp, validation_count, replace=True))
            id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][3].extend(np.random.choice(id_temp, validation_count, replace=True))

            
    RNA_test_count = int(0.2*len(RNA_id_list_r_only))
    ATAC_test_count = int(0.2*len(ATAC_id_list_a_only))
    
    for i in range(5):
        test_batch_r = RNA_id_list_r_only[0:RNA_test_count]
        test_batch_a = ATAC_id_list_a_only[0:ATAC_test_count]
        RNA_id_list_r_only.extend(test_batch_r)
        RNA_id_list_r_only = RNA_id_list_r_only[RNA_test_count:]
        ATAC_id_list_a_only.extend(test_batch_a)
        ATAC_id_list_a_only = ATAC_id_list_a_only[ATAC_test_count:]
        
        id_list[i][4].extend(test_batch_r)

        id_list[i][5].extend(test_batch_a)

            
            
    return id_list
