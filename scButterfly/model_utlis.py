import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import sparse
import anndata as ad
import pandas as pd
import gc


class RNA_ATAC_dataset(Dataset):
    def __init__(self, R_data, A_data, id_list_r, id_list_a):
        """
        Set random seed.

        Parameters
        ----------
        R_data
            complete RNA data for model traning and testing.
            
        A_data
            complete ATAC data for model traning and testing.

        id_list_r
            ids of cells in RNA data used for model training.
            
        id_list_a
            ids of cells in ATAC data used for model training.    
        
        """
        
        self.RNA_dataset = R_data        
        self.ATAC_dataset = A_data
        self.id_list_r = id_list_r
        self.id_list_a = id_list_a
        self.r_count = len(self.id_list_r)
        self.a_count = len(self.id_list_a)

    def __len__(self):
        return self.r_count

    def __getitem__(self, idx):
        RNA_x = self.RNA_dataset[self.id_list_r[idx], :]
        ATAC_x = self.ATAC_dataset[self.id_list_a[idx], :]
        RNA_x = torch.from_numpy(RNA_x)
        ATAC_x = torch.from_numpy(ATAC_x)
        sample = torch.cat([RNA_x, ATAC_x])
        return sample


class Single_omics_dataset(Dataset):
    def __init__(self, data, id_list):
        """
        Set random seed.

        Parameters
        ----------
        data
            complete data for model traning and testing.

        id_list
            id of cells used for model training.
        
        """

        self.dataset = data        
        self.id_list = id_list

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        x = self.dataset[self.id_list[idx], :]
        x = torch.from_numpy(x)
        return x   


def setup_seed(seed):
    """
    Set random seed.
    
    Parameters
    ----------
    seed
        Number to be set as random seed for reproducibility.
        
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    """Cite from https://github.com/Bjarten/early-stopping-pytorch"""
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save_model_dict(path) # save best model here
        self.val_loss_min = val_loss

        
def tensor2adata(x, val=1e-4):
    """
    Transform the tensor list to an Anndata.
    
    Parameters
    ----------
    x
        tensor list to concatenate.

    val
        value for sparse for matrix, default 1e-4.
        
    Returns
    ----------
    x
        concatenated anndata.
        
    """
    x = torch.cat(x)
    x = torch.masked_fill(x, x < val, 0)
    x = sparse.csr_matrix(x)
    x = ad.AnnData(x)
    return x


def tensor2adata_adt(x):
    """
    Transform the tensor list to an Anndata.
    
    Parameters
    ----------
    x
        tensor list to concatenate.

    Returns
    ----------
    x
        concatenated anndata.
        
    """
    x = torch.cat(x)
    x = sparse.csr_matrix(x)
    x = ad.AnnData(x)
    return x


def record_loss_log(
    pretrain_r_loss,
    pretrain_r_kl,
    pretrain_r_loss_val,
    pretrain_r_kl_val,
    pretrain_a_loss,
    pretrain_a_kl,
    pretrain_a_loss_val,
    pretrain_a_kl_val,
    train_loss,
    train_kl,
    train_discriminator,
    train_loss_val,
    train_kl_val,
    train_discriminator_val,
    path
):
    """
    Record run loss of training.
    
    Parameters
    ----------
    pretrain_r_loss
        train reconstruct loss of pretrain for RNA.
        
    pretrain_r_kl
        train kl divergence of pretrain for RNA.
        
    pretrain_r_loss_val
        validation reconstruct loss of pretrain for RNA.
        
    pretrain_r_kl_val
        validation kl divergence of pretrain for RNA.
        
    pretrain_a_loss
        train reconstruct loss of pretrain for ATAC.
        
    pretrain_a_kl
        train kl divergence of pretrain for ATAC.
        
    pretrain_a_loss_val
        validation reconstruct loss of pretrain for ATAC.
        
    pretrain_a_kl_val
        validation kl divergence of pretrain for ATAC.
        
    train_loss
        train reconstruct loss of train.
        
    train_kl
        train kl divergence of train.
        
    train_discriminator
        validation reconstruct loss of train.
        
    train_loss_val
        train reconstruct loss of train.
        
    train_kl_val
        train kl divergence of train.
        
    train_discriminator_val
        validation reconstruct train.
   
    path
        path for saving runlog.
    
    """
    fig_pretrain_r = plt.figure()
    ax_pretrain_r_re = fig_pretrain_r.add_subplot(1, 2, 1)
    ax_pretrain_r_kl = fig_pretrain_r.add_subplot(1, 2, 2)
    
    ax_pretrain_r_re.plot(pretrain_r_loss)
    ax_pretrain_r_re.plot(pretrain_r_loss_val)
    ax_pretrain_r_re.set_title('Reconstruct Loss for RNA Pretrain')
    
    ax_pretrain_r_kl.plot(pretrain_r_kl)
    ax_pretrain_r_kl.plot(pretrain_r_kl_val)
    ax_pretrain_r_kl.set_title('KL Divergence for RNA Pretrain')
    fig_pretrain_r.tight_layout()
    
    fig_pretrain_a = plt.figure()   
    ax_pretrain_a_re = fig_pretrain_a.add_subplot(1, 2, 1)
    ax_pretrain_a_kl = fig_pretrain_a.add_subplot(1, 2, 2)
    
    ax_pretrain_a_re.plot(pretrain_a_loss)
    ax_pretrain_a_re.plot(pretrain_a_loss_val)
    ax_pretrain_a_re.set_title('Reconstruct Loss for ATAC Pretrain')
    
    ax_pretrain_a_kl.plot(pretrain_a_kl)
    ax_pretrain_a_kl.plot(pretrain_a_kl_val)
    ax_pretrain_a_kl.set_title('KL Divergence for ATAC Pretrain')
    fig_pretrain_a.tight_layout()
    
    fig_train = plt.figure()
    ax_train_re = fig_train.add_subplot(1, 3, 1)
    ax_train_kl = fig_train.add_subplot(1, 3, 2)
    ax_train_dis = fig_train.add_subplot(1, 3, 3)
    
    ax_train_re.plot(train_loss)
    ax_train_re.plot(train_loss_val)
    ax_train_re.set_title('Reconstruct Loss for Train')
    
    ax_train_kl.plot(train_kl)
    ax_train_kl.plot(train_kl_val)
    ax_train_kl.set_title('KL Divergence for Train')
    
    ax_train_dis.plot(train_discriminator)
    ax_train_dis.plot(train_discriminator_val)
    ax_train_dis.set_title('Discriminator Loss for Train')
    fig_train.tight_layout()
    
    fig_list = [fig_pretrain_r, fig_pretrain_a, fig_train]
    with PdfPages(path + '/run_log.pdf') as pdf:
        for i in range(len(fig_list)):
            pdf.savefig(figure=fig_list[i], dpi=200)
            plt.close()
            

def idx2adata_multiVI(
    adata,
    train_id,
    validation_id, 
    test_id
):
    """
    Split datasets into train, validation and test part using cell id.
    
    Parameters
    ----------
    RNA_data
        full RNA data for spliting.
        
    ATAC_data
        full ATAC data for spliting.
        
    train_id
        cell index list used for model training.
   
    validation_id
        cell index list used for model validation.
        
    test_id
        cell index list used for model testing.
        
    """
    #train_id.extend(validation_id)
    train_adata = ad.AnnData(adata.X[train_id, :])
    test_adata = ad.AnnData(adata.X[test_id, :])

    train_adata.var = adata.var
    test_adata.var = adata.var

    train_adata.obs = adata.obs.iloc[train_id, :].copy()
    test_adata.obs = adata.obs.iloc[test_id, :].copy()

    train_adata.obs.reset_index(drop=True, inplace=True)
    test_adata.obs.reset_index(drop=True, inplace=True)

    return train_adata, test_adata