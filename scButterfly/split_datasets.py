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



def five_hold_split_dataset(
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


def celltype_split_dataset(
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


def unpaired_split_dataset_yao(
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

    cell_type_list = list(RNA_data.obs.cell_type.cat.categories)
    
    id_list = [[[] for j in range(6)] for i in range(4)]
    
    RNA_batch_list = list(RNA_data.obs.Donor.cat.categories)
    ATAC_batch_list = list(ATAC_data.obs.batch.cat.categories)
    
    random.shuffle(RNA_batch_list)
    random.shuffle(ATAC_batch_list)
    
    max_train_count = 500
    max_validation_count = 200
    
    for i in range(4):
        test_batch_r = RNA_batch_list[0:1]
        validation_batch_r = RNA_batch_list[1:2]
        train_batch_r = RNA_batch_list[2:4]
        test_batch_a = ATAC_batch_list[0:2]
        validation_batch_a = ATAC_batch_list[2:4]
        train_batch_a = ATAC_batch_list[4:9]
        RNA_batch_list.extend(test_batch_r)
        ATAC_batch_list.extend(test_batch_a)
        RNA_batch_list = RNA_batch_list[1:]
        ATAC_batch_list = ATAC_batch_list[2:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.Donor[RNA_data.obs.Donor == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.batch[ATAC_data.obs.batch == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.Donor == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.Donor == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        train_RNA = RNA_data.obs.loc[train_RNA_id, :]
        train_ATAC = ATAC_data.obs.loc[train_ATAC_id, :]
        validation_RNA = RNA_data.obs.loc[validation_RNA_id, :]
        validation_ATAC = ATAC_data.obs.loc[validation_ATAC_id, :]
        
        for ctype in cell_type_list:
            id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            id_list[i][0].extend(np.random.choice(id_temp, max_train_count, replace=True))     
            id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][1].extend(np.random.choice(id_temp, max_train_count, replace=True))
            id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            id_list[i][2].extend(np.random.choice(id_temp, max_validation_count, replace=True))
            id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][3].extend(np.random.choice(id_temp, max_validation_count, replace=True))
        
    return id_list


def unpaired_split_dataset_muto(
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

    cell_type_list = list(RNA_data.obs.cell_type.cat.categories)
    
    id_list = [[[] for j in range(6)] for i in range(5)]
    
    batch_list = list(RNA_data.obs.batch.cat.categories)
    
    random.shuffle(batch_list)
    
    max_train_count = 500
    max_validation_count = 200
    
    for i in range(5):
        test_batch_r = batch_list[0:1]
        validation_batch_r = batch_list[1:2]
        train_batch_r = batch_list[2:5]
        test_batch_a = batch_list[0:1]
        validation_batch_a = batch_list[1:2]
        train_batch_a = batch_list[2:5]
        batch_list.extend(test_batch_r)
        batch_list = batch_list[1:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.batch[RNA_data.obs.batch == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.batch[ATAC_data.obs.batch == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.batch == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.batch == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        train_RNA = RNA_data.obs.loc[train_RNA_id, :]
        train_ATAC = ATAC_data.obs.loc[train_ATAC_id, :]
        validation_RNA = RNA_data.obs.loc[validation_RNA_id, :]
        validation_ATAC = ATAC_data.obs.loc[validation_ATAC_id, :]
        
        for ctype in cell_type_list:
            id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            id_list[i][0].extend(np.random.choice(id_temp, max_train_count, replace=True))     
            id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][1].extend(np.random.choice(id_temp, max_train_count, replace=True))
            id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            id_list[i][2].extend(np.random.choice(id_temp, max_validation_count, replace=True))
            id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            id_list[i][3].extend(np.random.choice(id_temp, max_validation_count, replace=True))
        
    return id_list


def unpaired_split_dataset_yao_scglue(
    RNA_data, 
    ATAC_data, 
    combine_list,
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
    
    id_list = [[[] for j in range(6)] for i in range(4)]
    
    RNA_batch_list = list(RNA_data.obs.Donor.cat.categories)
    ATAC_batch_list = list(ATAC_data.obs.batch.cat.categories)
    
    random.shuffle(RNA_batch_list)
    random.shuffle(ATAC_batch_list)
    
    target_train = len(list(RNA_data.obs.cell_type.cat.categories)) * 500
    target_validation = len(list(RNA_data.obs.cell_type.cat.categories)) * 200
    
    for i in range(4):
        combine = combine_list[i]
        cell_type_list = list(combine.obs.leiden.cat.categories)
        combine_RNA = combine[combine.obs.domain == 'scRNA-seq']
        combine_ATAC = combine[combine.obs.domain == 'scATAC-seq']
        
        max_train_count = int(target_train / len(cell_type_list)) + 1
        max_validation_count = int(target_validation / len(cell_type_list)) + 1
        
        test_batch_r = RNA_batch_list[0:1]
        validation_batch_r = RNA_batch_list[1:2]
        train_batch_r = RNA_batch_list[2:4]
        test_batch_a = ATAC_batch_list[0:2]
        validation_batch_a = ATAC_batch_list[2:4]
        train_batch_a = ATAC_batch_list[4:9]
        RNA_batch_list.extend(test_batch_r)
        ATAC_batch_list.extend(test_batch_a)
        RNA_batch_list = RNA_batch_list[1:]
        ATAC_batch_list = ATAC_batch_list[2:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.Donor[RNA_data.obs.Donor == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.batch[ATAC_data.obs.batch == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.Donor == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.Donor == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        train_RNA = combine_RNA.obs.loc[train_RNA_id, :]
        train_ATAC = combine_ATAC.obs.loc[train_ATAC_id, :]
        validation_RNA = combine_RNA.obs.loc[validation_RNA_id, :]
        validation_ATAC = combine_ATAC.obs.loc[validation_ATAC_id, :]
        for ctype in cell_type_list:
            id_temp_train_RNA = list(train_RNA[train_RNA.leiden == ctype].index.astype(int))    
            id_temp_train_ATAC = list(train_ATAC[train_ATAC.leiden == ctype].index.astype(int))
            id_temp_validation_RNA = list(validation_RNA[validation_RNA.leiden == ctype].index.astype(int))
            id_temp_validation_ATAC = list(validation_ATAC[validation_ATAC.leiden == ctype].index.astype(int))
            if (len(id_temp_train_RNA) != 0) and (len(id_temp_train_ATAC) != 0):
                id_list[i][0].extend(np.random.choice(id_temp_train_RNA, max_train_count, replace=True))
                id_list[i][1].extend(np.random.choice(id_temp_train_ATAC, max_train_count, replace=True))
            if (len(id_temp_validation_RNA) != 0) and (len(id_temp_validation_ATAC) != 0):    
                id_list[i][2].extend(np.random.choice(id_temp_validation_RNA, max_validation_count, replace=True))
                id_list[i][3].extend(np.random.choice(id_temp_validation_ATAC, max_validation_count, replace=True))

    return id_list


def unpaired_split_dataset_muto_scglue(
    RNA_data, 
    ATAC_data, 
    combine_list,
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
    
    id_list = [[[] for j in range(6)] for i in range(5)]
    
    batch_list = list(RNA_data.obs.batch.cat.categories)
    
    random.shuffle(batch_list)
    
    target_train = len(list(RNA_data.obs.cell_type.cat.categories)) * 500
    target_validation = len(list(RNA_data.obs.cell_type.cat.categories)) * 200
    
    for i in range(5):
        combine = combine_list[i]
        cell_type_list = list(combine.obs.leiden.cat.categories)
        combine_RNA = combine[combine.obs.domain == 'scRNA-seq']
        combine_ATAC = combine[combine.obs.domain == 'scATAC-seq']
        
        max_train_count = int(target_train / len(cell_type_list)) + 1
        max_validation_count = int(target_validation / len(cell_type_list)) + 1

        test_batch_r = batch_list[0:1]
        validation_batch_r = batch_list[1:2]
        train_batch_r = batch_list[2:5]
        test_batch_a = batch_list[0:1]
        validation_batch_a = batch_list[1:2]
        train_batch_a = batch_list[2:5]
        batch_list.extend(test_batch_r)
        batch_list = batch_list[1:]
        
        for each in test_batch_r:
            id_list[i][4].extend(list(RNA_data.obs.batch[RNA_data.obs.batch == each].index.astype(int)))
            
        for each in test_batch_a:
            id_list[i][5].extend(list(ATAC_data.obs.batch[ATAC_data.obs.batch == each].index.astype(int)))
        
        train_RNA_id = []
        train_ATAC_id = []
        validation_RNA_id = []
        validation_ATAC_id = []
        for each in train_batch_r:
            train_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.batch == each].index))
        for each in train_batch_a:
            train_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        for each in validation_batch_r:
            validation_RNA_id.extend(list(RNA_data.obs[RNA_data.obs.batch == each].index))
        for each in validation_batch_a:
            validation_ATAC_id.extend(list(ATAC_data.obs[ATAC_data.obs.batch == each].index))
        train_RNA = combine_RNA.obs.loc[train_RNA_id, :]
        train_ATAC = combine_ATAC.obs.loc[train_ATAC_id, :]
        validation_RNA = combine_RNA.obs.loc[validation_RNA_id, :]
        validation_ATAC = combine_ATAC.obs.loc[validation_ATAC_id, :]
        for ctype in cell_type_list:
            id_temp_train_RNA = list(train_RNA[train_RNA.leiden == ctype].index.astype(int))    
            id_temp_train_ATAC = list(train_ATAC[train_ATAC.leiden == ctype].index.astype(int))
            id_temp_validation_RNA = list(validation_RNA[validation_RNA.leiden == ctype].index.astype(int))
            id_temp_validation_ATAC = list(validation_ATAC[validation_ATAC.leiden == ctype].index.astype(int))
            if (len(id_temp_train_RNA) != 0) and (len(id_temp_train_ATAC) != 0):
                id_list[i][0].extend(np.random.choice(id_temp_train_RNA, max_train_count, replace=True))
                id_list[i][1].extend(np.random.choice(id_temp_train_ATAC, max_train_count, replace=True))
            if (len(id_temp_validation_RNA) != 0) and (len(id_temp_validation_ATAC) != 0):    
                id_list[i][2].extend(np.random.choice(id_temp_validation_RNA, max_validation_count, replace=True))
                id_list[i][3].extend(np.random.choice(id_temp_validation_ATAC, max_validation_count, replace=True))
        
    return id_list


def seurat_batch_split_dataset(
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

    cell_type_list = list(RNA_data.obs.cell_type.cat.categories)
    
    id_list = [[[] for j in range(6)] for i in range(len(cell_type_list))]
    
    batch_list = list(RNA_data.obs.cell_type.cat.categories)
    
    random.shuffle(batch_list)
    
    max_train_count = 1200
    max_validation_count = 800
    
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
            max_train_count_temp = int(train_RNA.cell_type.value_counts(normalize=True)[ctype] * max_train_count)
            max_validation_count_temp = int(validation_RNA.cell_type.value_counts(normalize=True)[ctype] * max_validation_count)
            
            id_temp = list(train_RNA[train_RNA.cell_type == ctype].index.astype(int))
            if len(id_temp) != 0:
                id_list[i][0].extend(np.random.choice(id_temp, max_train_count_temp, replace=True))     
            id_temp = list(train_ATAC[train_ATAC.cell_type == ctype].index.astype(int))
            if len(id_temp) != 0:
                id_list[i][1].extend(np.random.choice(id_temp, max_train_count_temp, replace=True))
            id_temp = list(validation_RNA[validation_RNA.cell_type == ctype].index.astype(int))
            if len(id_temp) != 0:
                id_list[i][2].extend(np.random.choice(id_temp, max_validation_count_temp, replace=True))
            id_temp = list(validation_ATAC[validation_ATAC.cell_type == ctype].index.astype(int))
            if len(id_temp) != 0:
                id_list[i][3].extend(np.random.choice(id_temp, max_validation_count_temp, replace=True))
        
    return id_list