import os
import random
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scButterfly.model_component import *
from scButterfly.model_utlis import *
from scButterfly.calculate_cluster import *
from scButterfly.draw_cluster import *
from scButterfly.relation_evaluation import *
from scButterfly.data_processing import *
from scButterfly.logger import *


class Model():
    def __init__(self,
        RNA_data,
        ATAC_data,        
        chrom_list: list,
        logging_path: str,
        R_encoder_dim_list: list,
        A_encoder_dim_list: list,
        R_decoder_dim_list: list,
        A_decoder_dim_list: list,
        R_encoder_nlayer: int = 2, 
        A_encoder_nlayer: int = 2,
        R_decoder_nlayer: int = 2, 
        A_decoder_nlayer: int = 2,
        R_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list: list = [nn.LeakyReLU(), nn.Sigmoid()],
        translator_embed_dim: int = 128, 
        translator_input_dim_r: int = 128,
        translator_input_dim_a: int = 128,
        translator_embed_act_list: list = [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()],
        discriminator_nlayer: int = 1,
        discriminator_dim_list_R: list = [128],
        discriminator_dim_list_A: list = [128],
        discriminator_act_list: list = [nn.Sigmoid()],
        dropout_rate: float = 0.1,
        R_noise_rate: float = 0.5,
        A_noise_rate: float = 0,
    ):
        """
        Main model. Some parameters need information about data, please see in Tutorial.
        
        Parameters
        ----------
        RNA_data: Anndata
            RNA data for model training and testing.
            
        ATAC_data: Anndata
            ADT data for model training and testing.

        chrom_list: list
            list of peaks count for each chromosomes.
            
        logging_path: str
            the path for output process logging, if not save, set it None.
            
        R_encoder_dim_list: list
            dimension list of RNA encoder, length equal to R_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.
            
        A_encoder_dim_list: list
            dimension list of ADT encoder, length equal to A_encoder_nlayer + 1, the first equal to RNA data dimension, the last equal to embedding dimension.
            
        R_decoder_dim_list: list
            dimension list of RNA decoder, length equal to R_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.
            
        A_decoder_dim_list: list
            dimension list of ADT decoder, length equal to A_decoder_nlayer + 1, the last equal to embedding dimension, the first equal to RNA data dimension.
            
        R_encoder_nlayer: int
            layer counts of RNA encoder, default 2.
            
        A_encoder_nlayer: int
            layer counts of ADT encoder, default 2.
            
        R_decoder_nlayer: int
            layer counts of RNA decoder, default 2.
            
        A_decoder_nlayer: int
            layer counts of ADT decoder, default 2.
            
        R_encoder_act_list: list
            activation list of RNA encoder, length equal to R_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        A_encoder_act_list: list
            activation list of ADT encoder, length equal to A_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        R_decoder_act_list: list
            activation list of RNA decoder, length equal to R_decoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        A_decoder_act_list: list
            activation list of ADT decoder, length equal to A_decoder_nlayer, default [nn.LeakyReLU(), nn.Sigmoid()].
            
        translator_embed_dim: int
            dimension of embedding space for translator, default 128.
            
        translator_input_dim_r: int
            dimension of input from RNA encoder for translator, default 128.
            
        translator_input_dim_a: int
            dimension of input from ADT encoder for translator, default 128.
            
        translator_embed_act_list: list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation], default [nn.LeakyReLU(), nn.LeakyReLU(), nn.LeakyReLU()].
            
        discriminator_nlayer: int
            layer counts of discriminator, default 1.
            
        discriminator_dim_list_R: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_R, default [128].
            
        discriminator_dim_list_A: list
            dimension list of discriminator, length equal to discriminator_nlayer, the first equal to translator_input_dim_A, default [128].
            
        discriminator_act_list: list
            activation list of discriminator, length equal to  discriminator_nlayer, default [nn.Sigmoid()].
            
        dropout_rate: float
            rate of dropout for network, default 0.1.
       
        R_noise_rate: float
            rate of set part of RNA input data to 0, default 0.5.
            
        A_noise_rate: float
            rate of set part of ADT input data to 0, default 0.

        """
        if not logging_path is None:
            file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
            file_handle.writelines([
                '------------------------------\n'
                'Model Parameters\n'
                'R_encoder_nlayer: '+str(R_encoder_nlayer)+'\n', 
                'A_encoder_nlayer: '+str(A_encoder_nlayer)+'\n',
                'R_decoder_nlayer: '+str(R_decoder_nlayer)+'\n', 
                'A_decoder_nlayer: '+str(A_decoder_nlayer)+'\n',
                'R_encoder_dim_list: '+str(R_encoder_dim_list)+'\n',
                'A_encoder_dim_list: '+str(A_encoder_dim_list)+'\n',
                'R_decoder_dim_list: '+str(R_decoder_dim_list)+'\n',
                'A_decoder_dim_list: '+str(A_decoder_dim_list)+'\n',
                'R_encoder_act_list: '+str(R_encoder_act_list)+'\n',
                'A_encoder_act_list: '+str(A_encoder_act_list)+'\n',
                'R_decoder_act_list: '+str(R_decoder_act_list)+'\n',
                'A_decoder_act_list: '+str(A_decoder_act_list)+'\n',
                'translator_embed_dim: '+str(translator_embed_dim)+'\n', 
                'translator_input_dim_r: '+str(translator_input_dim_r)+'\n',
                'translator_input_dim_a: '+str(translator_input_dim_a)+'\n',
                'translator_embed_act_list: '+str(translator_embed_act_list)+'\n',
                'discriminator_nlayer: '+str(discriminator_nlayer)+'\n',
                'discriminator_dim_list_R: '+str(discriminator_dim_list_R)+'\n',
                'discriminator_dim_list_A: '+str(discriminator_dim_list_A)+'\n',
                'discriminator_act_list: '+str(discriminator_act_list)+'\n',
                'dropout_rate: '+str(dropout_rate)+'\n',
                'R_noise_rate: '+str(R_noise_rate)+'\n',
                'A_noise_rate: '+str(A_noise_rate)+'\n',
                'chrom_list: '+str(chrom_list)+'\n'
            ])
            file_handle.close()
        
        self.RNA_encoder = NetBlock(
            nlayer = R_encoder_nlayer,
            dim_list = R_encoder_dim_list,
            act_list = R_encoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = R_noise_rate)

        self.ATAC_encoder = NetBlock(
            nlayer = A_encoder_nlayer,
            dim_list = A_encoder_dim_list,
            act_list = A_encoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = A_noise_rate)
            
        self.RNA_decoder = NetBlock(
            nlayer = R_decoder_nlayer,
            dim_list = R_decoder_dim_list,
            act_list = R_decoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = 0)
            
        self.ATAC_decoder = NetBlock(
            nlayer = A_decoder_nlayer,
            dim_list = A_decoder_dim_list,
            act_list = A_decoder_act_list,
            dropout_rate = dropout_rate,
            noise_rate = 0)
        
        self.R_translator = Single_Translator(
            translator_input_dim = translator_input_dim_r, 
            translator_embed_dim = translator_embed_dim, 
            translator_embed_act_list = translator_embed_act_list)
        
        self.A_translator = Single_Translator(
            translator_input_dim = translator_input_dim_a, 
            translator_embed_dim = translator_embed_dim, 
            translator_embed_act_list = translator_embed_act_list)
        
        self.translator = Translator(
            translator_input_dim_r = translator_input_dim_r, 
            translator_input_dim_a = translator_input_dim_a, 
            translator_embed_dim = translator_embed_dim, 
            translator_embed_act_list = translator_embed_act_list)

        discriminator_dim_list_R.append(1)
        discriminator_dim_list_A.append(1)
        self.discriminator_R = NetBlock(
            nlayer = discriminator_nlayer,
            dim_list = discriminator_dim_list_R,
            act_list = discriminator_act_list,
            dropout_rate = 0,
            noise_rate = 0)

        self.discriminator_A = NetBlock(
            nlayer = discriminator_nlayer,
            dim_list = discriminator_dim_list_A,
            act_list = discriminator_act_list,
            dropout_rate = 0,
            noise_rate = 0)
        
        # use GPU for training if cuda is available
        if torch.cuda.is_available():
            self.RNA_encoder = self.RNA_encoder.cuda()
            self.RNA_decoder = self.RNA_decoder.cuda()
            self.ATAC_encoder = self.ATAC_encoder.cuda()
            self.ATAC_decoder = self.ATAC_decoder.cuda()
            self.R_translator = self.R_translator.cuda()
            self.A_translator = self.A_translator.cuda()
            self.translator = self.translator.cuda()
            self.discriminator_R = self.discriminator_R.cuda()
            self.discriminator_A = self.discriminator_A.cuda()
        
        self.is_train_finished = False
        self.RNA_data_obs = RNA_data.obs
        self.ATAC_data_obs = ATAC_data.obs
        self.RNA_data_var = RNA_data.var
        self.ATAC_data_var = RNA_data.var
        self.RNA_data = RNA_data.X.toarray()
        self.ATAC_data = ATAC_data.X.toarray()
        self.num_workers = 0
        
    def set_train(self):
        self.RNA_encoder.train()
        self.RNA_decoder.train()
        self.ATAC_encoder.train()
        self.ATAC_decoder.train()
        self.R_translator.train()
        self.A_translator.train()
        self.translator.train()
        self.discriminator_R.train()
        self.discriminator_A.train()
     
    
    def set_eval(self):
        self.RNA_encoder.eval()
        self.RNA_decoder.eval()
        self.ATAC_encoder.eval()
        self.ATAC_decoder.eval()
        self.R_translator.eval()
        self.A_translator.eval()        
        self.translator.eval()
        self.discriminator_R.eval()
        self.discriminator_A.eval()
    
        
    def forward_R2R(self, RNA_input, r_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.R_translator(self.RNA_encoder(RNA_input), forward_type)
        predict_RNA = self.RNA_decoder(latent_layer)
        reconstruct_loss = r_loss(predict_RNA, RNA_input)
        kl_div_r = - 0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_r
        return loss, reconstruct_loss, kl_div_r
    
    
    def forward_A2A(self, ATAC_input, a_loss, kl_div_w, forward_type):
        latent_layer, mu, d = self.A_translator(self.ATAC_encoder(ATAC_input), forward_type)
        predict_ATAC = self.ATAC_decoder(latent_layer)
        reconstruct_loss = a_loss(predict_ATAC, ATAC_input)
        kl_div_a = - 0.5 * torch.mean(1 + d - mu.pow(2) - d.exp())
        loss = reconstruct_loss + kl_div_w * kl_div_a
        return loss, reconstruct_loss, kl_div_a
        
       
    def forward_translator(self, batch_samples, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, loss_weight, forward_type, kl_div_mean=False):
    
        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
    
        # forward generator
        
        R2 = self.RNA_encoder(RNA_input)
        A2 = self.ATAC_encoder(ATAC_input)
        if forward_type == 'train':
            R2R, R2A, mu_r, sigma_r = self.translator.train_model(R2, 'RNA')
            A2R, A2A, mu_a, sigma_a = self.translator.train_model(A2, 'ATAC')
        elif forward_type == 'test':
            R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
            A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')        
        
        R2R = self.RNA_decoder(R2R)
        R2A = self.ATAC_decoder(R2A)
        A2R = self.RNA_decoder(A2R)
        A2A = self.ATAC_decoder(A2A)
        
        # reconstruct loss
        lossR2R = r_loss(R2R, RNA_input)
        lossA2R = r_loss(A2R, RNA_input)
        lossR2A = a_loss(R2A, ATAC_input)
        lossA2A = a_loss(A2A, ATAC_input)
        
        # kl divergence
        if kl_div_mean:
            kl_div_r = - 0.5 * torch.mean(1 + sigma_r - mu_r.pow(2) - sigma_r.exp())
            kl_div_a = - 0.5 * torch.mean(1 + sigma_a - mu_a.pow(2) - sigma_a.exp())
        else:
            kl_div_r = torch.clamp(- 0.5 * torch.sum(1 + sigma_r - mu_r.pow(2) - sigma_r.exp()), 0, 10000)
            kl_div_a = torch.clamp(- 0.5 * torch.sum(1 + sigma_a - mu_a.pow(2) - sigma_a.exp()), 0, 10000)
        
        # calculate the loss
        r_loss_w, a_loss_w, d_loss_w, kl_div_R, kl_div_A, kl_div_w = loss_weight
        reconstruct_loss = r_loss_w * (lossR2R + lossA2R) + a_loss_w * (lossR2A + lossA2A)

        kl_div = kl_div_r + kl_div_a
        
        loss_g = kl_div_w * kl_div + reconstruct_loss

        return reconstruct_loss, kl_div, loss_g
    
    
    def forward_discriminator(self, batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, forward_type):
        
        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

        # forward of generator
        R2 = self.RNA_encoder(RNA_input)
        A2 = self.ATAC_encoder(ATAC_input)
        if forward_type == 'train':
            R2R, R2A, mu_r, sigma_r = self.translator.train_model(R2, 'RNA')
            A2R, A2A, mu_a, sigma_a = self.translator.train_model(A2, 'ATAC')
        elif forward_type == 'test':
            R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
            A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')

        batch_size = batch_samples.shape[0]

        # 1 menas a real data 0 menas a generated data, here use a soft label
        temp1 = np.random.rand(batch_size)
        temp = [0 for item in temp1]
        for i in range(len(temp1)):
            if temp1[i] > 0.8:
                temp[i] = temp1[i]
            elif temp1[i] <= 0.8 and temp1[i] > 0.5:
                temp[i] = 0.8
            elif temp1[i] <= 0.5 and temp1[i] > 0.2:
                temp[i] = 0.2
            else:
                temp[i] = temp1[i]

        input_data_a = torch.stack([A2[i] if temp[i] > 0.5 else R2A[i] for i in range(batch_size)], dim=0)
        input_data_r = torch.stack([R2[i] if temp[i] > 0.5 else A2R[i] for i in range(batch_size)], dim=0)

        predict_atac = self.discriminator_A(input_data_a)
        predict_rna = self.discriminator_R(input_data_r)

        loss1 = d_loss(predict_atac.reshape(batch_size), torch.tensor(temp).cuda().float())
        loss2 = d_loss(predict_rna.reshape(batch_size), torch.tensor(temp).cuda().float())
        return loss1 + loss2
        
    
    def save_model_dict(self, output_path):
        torch.save(self.RNA_encoder.state_dict(), output_path + '/model/RNA_encoder.pt')
        torch.save(self.ATAC_encoder.state_dict(), output_path + '/model/ATAC_encoder.pt')
        torch.save(self.RNA_decoder.state_dict(), output_path + '/model/RNA_decoder.pt')
        torch.save(self.ATAC_decoder.state_dict(), output_path + '/model/ATAC_decoder.pt')
        torch.save(self.R_translator.state_dict(), output_path + '/model/R_translator.pt')
        torch.save(self.A_translator.state_dict(), output_path + '/model/A_translator.pt')
        torch.save(self.translator.state_dict(), output_path + '/model/translator.pt')
        torch.save(self.discriminator_A.state_dict(), output_path + '/model/discriminator_A.pt')
        torch.save(self.discriminator_R.state_dict(), output_path + '/model/discriminator_R.pt')
   

    def evaluation(
        self,
        max_evaluate : int,
        RNA_data,
        ATAC_data,
        id_list
    ):
        mean_test_loss = 0
        mean_test_auroc = 0
        mean_test_aupr = 0
        mean_test_pearson = 0
        mean_test_spearman = 0
        sum_auroc = 0
        sum_aupr = 0
        max_enum = 9
        RNA_input_dim = RNA_data.X.toarray().shape[1]
        ATAC_input_dim = ATAC_data.X.toarray().shape[1]
        
        self.test_dataset = RNA_ATAC_dataset(RNA_data, ATAC_data, id_list, id_list)
        
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
        
        self.set_eval()
        batch_samples_list_A = []
        # to get every_mean
        for epoch1 in range(1):
            with torch.no_grad():
                for idx, batch_samples in enumerate(self.test_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
                    for i in range(len(ATAC_input)):
                        batch_samples_list_A.append(torch.reshape(ATAC_input[i].cpu(),(1,len(ATAC_input[0]))))
            batch_samples_list_A = torch.cat(batch_samples_list_A, dim = 0)
            every_mean = torch.mean(batch_samples_list_A, dim = 0)
        
        with torch.no_grad():
            for idx, batch_samples in enumerate(self.test_dataloader):
                if torch.cuda.is_available():
                    batch_samples = batch_samples.cuda().to(torch.float32)


                RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
                R2 = self.RNA_encoder(RNA_input)
                A2 = self.ATAC_encoder(ATAC_input)
                R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
                A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')
                R2A = self.ATAC_decoder(R2A)
                A2R = self.RNA_decoder(A2R)

                atac_input = ATAC_input.cpu().numpy()
                rna_input = RNA_input.cpu().numpy()
                r2a = R2A.cpu().detach().numpy()
                a2r = A2R.cpu().detach().numpy()
                ss = sum(atac_input)

                for i in range(len(atac_input)):
                    a=AUROC(r2a[i,:],atac_input[i,:], every_mean)
                    if a != 0:
                        mean_test_auroc += a
                        sum_auroc += 1
                    else:
                        mean_test_auroc += a
                    b = AUPR_norm(r2a[i,:],atac_input[i,:], every_mean)
                    if b!= 0:
                        mean_test_aupr += b
                        sum_aupr += 1
                    else:
                        mean_test_aupr += b

                    mean_test_pearson += pearson(a2r[i,:],rna_input[i,:])
                    mean_test_spearman += spearman(a2r[i,:],rna_input[i,:])
                print(idx)
                        
                if idx == max_evaluate-1:
                    break

        mean_test_auroc /= sum_auroc
        mean_test_aupr /= sum_aupr
        mean_test_pearson /= max_evaluate
        mean_test_spearman /= max_evaluate
 
        return [mean_test_auroc,mean_test_aupr,mean_test_pearson,mean_test_spearman]



    def train(
        self,
        loss_weight: list,
        train_id_r: list,
        train_id_a: list,
        validation_id_r: list,
        validation_id_a: list,
        R_encoder_lr: float = 0.001,
        A_encoder_lr: float = 0.001,
        R_decoder_lr: float = 0.001,
        A_decoder_lr: float = 0.001,
        R_translator_lr: float = 0.001,
        A_translator_lr: float = 0.001,
        translator_lr: float = 0.001,
        discriminator_lr: float = 0.005,
        R2R_pretrain_epoch: int = 100,
        A2A_pretrain_epoch: int = 100,
        lock_encoder_and_decoder: bool = False,
        translator_epoch: int = 200,
        patience: int = 50,
        batch_size: int = 64,
        r_loss = nn.MSELoss(size_average=True),
        a_loss = nn.MSELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        output_path: str = None,
        seed: int = 19193,
        kl_mean: bool = True,
        R_pretrain_kl_warmup: int = 50,
        A_pretrain_kl_warmup: int = 50,
        translation_kl_warmup: int = 50,
        load_model: str = None,
        logging_path: str = None
    ):
        """
        Training for model. Some parameters need information about data, please see in Tutorial.
        
        Parameters
        ----------
        loss_weight: list
            list of loss weight for [r_loss, a_loss, d_loss, kl_div_R, kl_div_A, kl_div_all].
        
        train_id_r: list
            list of RNA data cell ids for training.
            
        train_id_a: list
            list of ADT data cell ids for training.
            
        validation_id_r: list
            list of RNA data cell ids for validation.
        
        validation_id_a: list
            list of ADT data cell ids for validation.

        R_encoder_lr: float
            learning rate of RNA encoder, default 0.001.
            
        A_encoder_lr: float
            learning rate of ADT encoder, default 0.001.
            
        R_decoder_lr: float
            learning rate of RNA decoder, default 0.001.
            
        A_decoder_lr: float
            learning rate of ADT decoder, default 0.001.
       
        R_translator_lr: float
            learning rate of RNA pretrain translator, default 0.001.
            
        A_translator_lr: float
            learning rate of ADT pretrain translator, default 0.001.
            
        translator_lr: float
            learning rate of translator, default 0.001.
            
        discriminator_lr: float
            learning rate of discriminator, default 0.005.
            
        R2R_pretrain_epoch: int
            max epoch for pretrain RNA autoencoder, default 100.
            
        A2A_pretrain_epoch: int
            max epoch for pretrain ADT autoencoder, default 100.
            
        lock_encoder_and_decoder: bool
            lock the pretrained encoder and decoder or not, default False.
            
        translator_epoch: int
            max epoch for train translator, default 200.
            
        patience: int
            patience for loss on validation, default 50.
            
        batch_size: int
            batch size for training and validation, default 64.
            
        r_loss
            loss function for RNA reconstruction, default nn.MSELoss(size_average=True).
            
        a_loss
            loss function for ADT reconstruction, default nn.MSELoss(size_average=True).
            
        d_loss
            loss function for discriminator, default nn.BCELoss(size_average=True).
            
        output_path: str
            file path for model output, default None.
            
        seed: int
            set up the random seed, default 19193.
            
        kl_mean: bool
            size average for kl divergence or not, default True.
            
        R_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in RNA pretrain, default 50.
        
        A_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in ADT pretrain, default 50.
        
        translation_kl_warmup: int
            epoch of linear weight warm up for kl divergence in translator pretrain, default 50.
            
        load_model: str
            the path for loading model if needed, else set it None, default None.
            
        logging_path: str
            the path for output process logging, if not save, set it None, default None.
            
        """
        if not logging_path is None:
            file_handle=open(logging_path + '/Parameters_Record.txt',mode='a')
            file_handle.writelines([
                '------------------------------\n'
                'Train Parameters\n'
                'R_encoder_lr: '+str(R_encoder_lr)+'\n',
                'A_encoder_lr: '+str(A_encoder_lr)+'\n',
                'R_decoder_lr: '+str(R_decoder_lr)+'\n',
                'A_decoder_lr: '+str(A_decoder_lr)+'\n',
                'R_translator_lr: '+str(R_translator_lr)+'\n',
                'A_translator_lr: '+str(A_translator_lr)+'\n',
                'translator_lr: '+str(translator_lr)+'\n',
                'discriminator_lr: '+str(discriminator_lr)+'\n',
                'R2R_pretrain_epoch: '+str(R2R_pretrain_epoch)+'\n',
                'A2A_pretrain_epoch: '+str(A2A_pretrain_epoch)+'\n',
                'lock_encoder_and_decoder: '+str(lock_encoder_and_decoder)+'\n',
                'translator_epoch: '+str(translator_epoch)+'\n',
                'patience: '+str(patience)+'\n',
                'batch_size: '+str(batch_size)+'\n',
                'r_loss: '+str(r_loss)+'\n',
                'a_loss: '+str(a_loss)+'\n',
                'd_loss: '+str(d_loss)+'\n',
                'loss_weight: '+str(loss_weight)+'\n',
                'seed: '+str(seed)+'\n',
                'kl_mean: '+str(kl_mean)+'\n',
                'R_pretrain_kl_warmup: '+str(R_pretrain_kl_warmup)+'\n',
                'A_pretrain_kl_warmup: '+str(A_pretrain_kl_warmup)+'\n',
                'translation_kl_warmup: '+str(translation_kl_warmup)+'\n',
                'load_model: '+str(load_model)+'\n'
            ])
            file_handle.close()
            
        my_logger = create_logger(name='Trainer', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        
        self.is_train_finished = False
        
        if output_path is None:
            output_path = '.'
        
        if not load_model is None:
            my_logger.info('load pretrained model from path: '+str(load_model)+'/model/')
            self.RNA_encoder.load_state_dict(torch.load(load_model + '/model/RNA_encoder.pt'))
            self.ATAC_encoder.load_state_dict(torch.load(load_model + '/model/ATAC_encoder.pt'))
            self.RNA_decoder.load_state_dict(torch.load(load_model + '/model/RNA_decoder.pt'))
            self.ATAC_decoder.load_state_dict(torch.load(load_model + '/model/ATAC_decoder.pt'))
            self.translator.load_state_dict(torch.load(load_model + '/model/translator.pt'))
            self.discriminator_A.load_state_dict(torch.load(load_model + '/model/discriminator_A.pt'))
            self.discriminator_R.load_state_dict(torch.load(load_model + '/model/discriminator_R.pt'))
            
        if not seed is None:
            setup_seed(seed)
        
        RNA_input_dim = self.RNA_data.shape[1]
        ATAC_input_dim = self.ATAC_data.shape[1]
        cell_count = len(train_id_r)
        
        # define the dataset and dataloader for train and validation
        self.train_dataset = RNA_ATAC_dataset(self.RNA_data, self.ATAC_data, train_id_r, train_id_a)
        self.validation_dataset = RNA_ATAC_dataset(self.RNA_data, self.ATAC_data, validation_id_r, validation_id_a)

        if cell_count % batch_size == 1:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)
        else:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.validation_dataloader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        self.optimizer_R_encoder = torch.optim.Adam(self.RNA_encoder.parameters(), lr=R_encoder_lr)
        self.optimizer_A_encoder = torch.optim.Adam(self.ATAC_encoder.parameters(), lr=A_encoder_lr, weight_decay=0)
        self.optimizer_R_decoder = torch.optim.Adam(self.RNA_decoder.parameters(), lr=R_decoder_lr)
        self.optimizer_A_decoder = torch.optim.Adam(self.ATAC_decoder.parameters(), lr=A_decoder_lr, weight_decay=0)
        self.optimizer_R_translator = torch.optim.Adam(self.R_translator.parameters(), lr=R_translator_lr)
        self.optimizer_A_translator = torch.optim.Adam(self.A_translator.parameters(), lr=A_translator_lr)
        self.optimizer_translator = torch.optim.Adam(self.translator.parameters(), lr=translator_lr)
        self.optimizer_discriminator_A = torch.optim.SGD(self.discriminator_A.parameters(), lr=discriminator_lr)
        self.optimizer_discriminator_R = torch.optim.SGD(self.discriminator_R.parameters(), lr=discriminator_lr)
        
        """ eraly stop for model """
        self.early_stopping_R2R = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_A2A = EarlyStopping(patience=patience, verbose=False)
        self.early_stopping_all = EarlyStopping(patience=patience, verbose=False)
        
        if not os.path.exists(output_path + '/model'):
            os.mkdir(output_path + '/model')
        
        """ pretrain for RNA and ADT """
        my_logger.info('RNA pretraining ...')
        pretrain_r_loss, pretrain_r_kl, pretrain_r_loss_val, pretrain_r_kl_val = [], [], [], []
        with tqdm(total = R2R_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description('RNA pretrain')
            for epoch in range(R2R_pretrain_epoch):
                pretrain_r_loss_, pretrain_r_kl_, pretrain_r_loss_val_, pretrain_r_kl_val_ = [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader): 

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    """ pretrain for RNA """
                    weight_temp = loss_weight.copy()
                    if epoch < R_pretrain_kl_warmup:
                        weight_temp[3] = loss_weight[3] * epoch / R_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(RNA_input, r_loss, weight_temp[3], 'train')
                    self.optimizer_R_encoder.zero_grad()
                    self.optimizer_R_decoder.zero_grad()
                    self.optimizer_R_translator.zero_grad()
                    loss.backward()
                    self.optimizer_R_encoder.step()
                    self.optimizer_R_decoder.step()
                    self.optimizer_R_translator.step()

                    pretrain_r_loss_.append(reconstruct_loss.item())
                    pretrain_r_kl_.append(kl_div_r.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    loss, reconstruct_loss, kl_div_r = self.forward_R2R(RNA_input, r_loss, weight_temp[3], 'test')

                    pretrain_r_loss_val_.append(reconstruct_loss.item())
                    pretrain_r_kl_val_.append(kl_div_r.item())

                pretrain_r_loss.append(np.mean(pretrain_r_loss_))
                pretrain_r_kl.append(np.mean(pretrain_r_kl_))
                pretrain_r_loss_val.append(np.mean(pretrain_r_loss_val_))
                pretrain_r_kl_val.append(np.mean(pretrain_r_kl_val_))

                self.early_stopping_R2R(np.mean(pretrain_r_loss_val_), self, output_path)
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(pretrain_r_loss_val_)), 
                    val='{:.4f}'.format(np.mean(pretrain_r_loss_)))
                
                if self.early_stopping_R2R.early_stop:
                    my_logger.info('RNA pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                    self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                    self.R_translator.load_state_dict(torch.load(output_path + '/model/R_translator.pt'))
                    break
        
        
        pretrain_a_loss, pretrain_a_kl, pretrain_a_loss_val, pretrain_a_kl_val = [], [], [], []
        my_logger.info('ADT pretraining ...')
        with tqdm(total = A2A_pretrain_epoch, ncols=100) as pbar:
            pbar.set_description('ADT pretrain')
            for epoch in range(A2A_pretrain_epoch):
                pretrain_a_loss_, pretrain_a_kl_, pretrain_a_loss_val_, pretrain_a_kl_val_ = [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader): 

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    """ pretrain for ADT """
                    weight_temp = loss_weight.copy()
                    if epoch < A_pretrain_kl_warmup:
                        weight_temp[4] = loss_weight[4] * epoch / A_pretrain_kl_warmup

                    loss, reconstruct_loss, kl_div_a = self.forward_A2A(ATAC_input, a_loss, weight_temp[4], 'train')
                    self.optimizer_A_encoder.zero_grad()
                    self.optimizer_A_decoder.zero_grad()
                    self.optimizer_A_translator.zero_grad()
                    loss.backward()
                    self.optimizer_A_encoder.step()
                    self.optimizer_A_decoder.step()
                    self.optimizer_A_translator.step()

                    pretrain_a_loss_.append(reconstruct_loss.item())
                    pretrain_a_kl_.append(kl_div_a.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    loss, reconstruct_loss, kl_div_a = self.forward_A2A(ATAC_input, a_loss, weight_temp[4], 'test')

                    pretrain_a_loss_val_.append(reconstruct_loss.item())
                    pretrain_a_kl_val_.append(kl_div_a.item())

                pretrain_a_loss.append(np.mean(pretrain_a_loss_))
                pretrain_a_kl.append(np.mean(pretrain_a_kl_))
                pretrain_a_loss_val.append(np.mean(pretrain_a_loss_val_))
                pretrain_a_kl_val.append(np.mean(pretrain_a_kl_val_))

                self.early_stopping_A2A(np.mean(pretrain_a_loss_val_), self, output_path)
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(pretrain_a_loss_val_)), 
                    val='{:.4f}'.format(np.mean(pretrain_a_loss_)))
                
                if self.early_stopping_A2A.early_stop:
                    my_logger.info('ADT pretraining early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.ATAC_encoder.load_state_dict(torch.load(output_path + '/model/ATAC_encoder.pt'))
                    self.ATAC_decoder.load_state_dict(torch.load(output_path + '/model/ATAC_decoder.pt'))
                    self.A_translator.load_state_dict(torch.load(output_path + '/model/A_translator.pt'))
                    break

        
        """ train for translator and discriminator """
        train_loss, train_kl, train_discriminator, train_loss_val, train_kl_val, train_discriminator_val = [], [], [], [], [], []
        my_logger.info('Combine training ...')
        with tqdm(total = translator_epoch, ncols=100) as pbar:
            pbar.set_description('Combine training')
            for epoch in range(translator_epoch):
                train_loss_, train_kl_, train_discriminator_, train_loss_val_, train_kl_val_, train_discriminator_val_ = [], [], [], [], [], []
                self.set_train()
                for idx, batch_samples in enumerate(self.train_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    """ train for discriminator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, 'train')
                    self.optimizer_discriminator_R.zero_grad()
                    self.optimizer_discriminator_A.zero_grad()
                    loss_d.backward()
                    self.optimizer_discriminator_R.step()
                    self.optimizer_discriminator_A.step()

                    """ train for generator """
                    weight_temp = loss_weight.copy()
                    if epoch < translation_kl_warmup:
                        weight_temp[5] = loss_weight[5] * epoch / translation_kl_warmup
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, 'train')
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(batch_samples, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train', kl_mean)

                    if loss_d.item() < 1.35:
                        loss_g -= loss_weight[2] * loss_d

                    self.optimizer_translator.zero_grad()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.zero_grad()
                        self.optimizer_A_encoder.zero_grad()
                        self.optimizer_R_decoder.zero_grad()
                        self.optimizer_A_decoder.zero_grad()
                    loss_g.backward()
                    self.optimizer_translator.step()
                    if not lock_encoder_and_decoder:
                        self.optimizer_R_encoder.step()
                        self.optimizer_A_encoder.step()
                        self.optimizer_R_decoder.step()
                        self.optimizer_A_decoder.step()

                    train_loss_.append(reconstruct_loss.item())
                    train_kl_.append(kl_div.item()) 
                    train_discriminator_.append(loss_d.item())

                self.set_eval()
                for idx, batch_samples in enumerate(self.validation_dataloader):

                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)                    

                    RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)

                    """ test for discriminator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, 'test')

                    """ test for generator """
                    loss_d = self.forward_discriminator(batch_samples, RNA_input_dim, ATAC_input_dim, d_loss, 'test')
                    reconstruct_loss, kl_div, loss_g = self.forward_translator(batch_samples, RNA_input_dim, ATAC_input_dim, a_loss, r_loss, weight_temp, 'train', kl_mean)
                    loss_g -= loss_weight[2] * loss_d

                    train_loss_val_.append(reconstruct_loss.item())
                    train_kl_val_.append(kl_div.item()) 
                    train_discriminator_val_.append(loss_d.item())

                train_loss.append(np.mean(train_loss_))
                train_kl.append(np.mean(train_kl_))
                train_discriminator.append(np.mean(train_discriminator_))
                train_loss_val.append(np.mean(train_loss_val_))
                train_kl_val.append(np.mean(train_kl_val_))
                train_discriminator_val.append(np.mean(train_discriminator_val_))
                self.early_stopping_all(np.mean(train_loss_val_), self, output_path)
                
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(train_loss_val_)), 
                    val='{:.4f}'.format(np.mean(train_loss_)))
                
                if self.early_stopping_all.early_stop:
                    my_logger.info('Combine training early stop, validation loss does not improve in '+str(patience)+' epoches!')
                    self.RNA_encoder.load_state_dict(torch.load(output_path + '/model/RNA_encoder.pt'))
                    self.ATAC_encoder.load_state_dict(torch.load(output_path + '/model/ATAC_encoder.pt'))
                    self.RNA_decoder.load_state_dict(torch.load(output_path + '/model/RNA_decoder.pt'))
                    self.ATAC_decoder.load_state_dict(torch.load(output_path + '/model/ATAC_decoder.pt'))
                    self.translator.load_state_dict(torch.load(output_path + '/model/translator.pt'))
                    self.discriminator_A.load_state_dict(torch.load(output_path + '/model/discriminator_A.pt'))
                    self.discriminator_R.load_state_dict(torch.load(output_path + '/model/discriminator_R.pt'))
                    break
        
        self.save_model_dict(output_path)
            
        self.is_train_finished = True
        
        record_loss_log(
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
            output_path
        )

    def test(
        self,
        test_id_r: list,
        test_id_a: list,
        model_path: str = None,
        load_model: bool = True,
        output_path: str = None,
        test_evaluate: bool = True,
        test_cluster: bool = True,
        test_figure: bool = True,
        output_data: bool = False,
        return_predict: bool = False
    ):
        """
        Test for model.
        
        Parameters
        ----------            
        train_id_r: list
            list of RNA data cell ids for training.
            
        train_id_a: list
            list of ATAC data cell ids for training.
            
        model_path: str
            path for load trained model, default None.
            
        load_model: bool
            load the pretrained model or not, deafult True.
            
        output_path: str
            file path for model output, default None.
            
        test_evaludate: bool
            test correlation or not, deafult True.
            
        test_cluster: bool
            test clustrer index or not, deafult True.
            
        test_figure: bool
            test tSNE or not, deafult True.
            
        output_data: bool
            output the predicted test data to file or not, deafult False.
            
        return_predict: bool
            return predict or not, if True, output (A2R_predict, R2A_predict) as returns, deafult False.
            
        """
        """ load model from model_path if need """
        my_logger = create_logger(name='Tester', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
        
        if output_path is None:
            output_path = '.'
        
        if load_model:
            my_logger.info('load trained model from path: '+str(model_path)+'/model')
            self.RNA_encoder.load_state_dict(torch.load(model_path + '/model/RNA_encoder.pt'))
            self.ATAC_encoder.load_state_dict(torch.load(model_path + '/model/ATAC_encoder.pt'))
            self.RNA_decoder.load_state_dict(torch.load(model_path + '/model/RNA_decoder.pt'))
            self.ATAC_decoder.load_state_dict(torch.load(model_path + '/model/ATAC_decoder.pt'))
            self.translator.load_state_dict(torch.load(model_path + '/model/translator.pt'))
        
        """ load data """
        RNA_input_dim = self.RNA_data.shape[1]
        ATAC_input_dim = self.ATAC_data.shape[1]
        
        self.R_test_dataset = Single_omics_dataset(self.RNA_data, test_id_r)
        self.A_test_dataset = Single_omics_dataset(self.ATAC_data, test_id_a)
        self.R_test_dataloader = DataLoader(self.R_test_dataset, batch_size=64, shuffle=False, num_workers=self.num_workers)
        self.A_test_dataloader = DataLoader(self.A_test_dataset, batch_size=64, shuffle=False, num_workers=self.num_workers)

        self.set_eval()
        my_logger.info('get predicting ...')
        """ record the predicted data """
        R2A_predict = []
        A2R_predict = []
        R2_predict = []
        A2_predict = []
        with torch.no_grad():
            with tqdm(total = len(self.R_test_dataloader), ncols=100) as pbar:
                pbar.set_description('RNA to ADT predicting...')
                for idx, batch_samples in enumerate(self.R_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    R2 = self.RNA_encoder(batch_samples)
                    R2R, R2A, mu_r, sigma_r = self.translator.test_model(R2, 'RNA')
                    R2A = self.ATAC_decoder(R2A)
                    
                    R2A_predict.append(R2A.cpu())
                    R2_predict.append(mu_r.cpu())
                    
                    time.sleep(0.01)
                    pbar.update(1)

        with torch.no_grad():
            with tqdm(total = len(self.A_test_dataloader), ncols=100) as pbar:
                pbar.set_description('ADT to RNA predicting...')
                for idx, batch_samples in enumerate(self.A_test_dataloader):
                    if torch.cuda.is_available():
                        batch_samples = batch_samples.cuda().to(torch.float32)

                    A2 = self.ATAC_encoder(batch_samples)
                    A2R, A2A, mu_a, sigma_a = self.translator.test_model(A2, 'ATAC')
                    A2R = self.RNA_decoder(A2R)

                    A2R_predict.append(A2R.cpu())                       
                    A2_predict.append(mu_a.cpu())
                    
                    time.sleep(0.01)
                    pbar.update(1)
        
        R2A_predict = tensor2adata(R2A_predict)
        A2R_predict = tensor2adata(A2R_predict)
        
        A2R_predict.obs = self.ATAC_data_obs.iloc[test_id_a, :]
        R2A_predict.obs = self.RNA_data_obs.iloc[test_id_r, :]

        """ draw tsne if needed """
        if test_figure:
            my_logger.info('drawing tsne figures ...')
            fig_A2R = draw_tsne(A2R_predict, 'a2r', 'cell_type')
            fig_R2A = draw_tsne(R2A_predict, 'r2a', 'cell_type')
            fig_list = [fig_A2R, fig_R2A]
            with PdfPages(output_path + '/tSNE.pdf') as pdf:
                for i in range(len(fig_list)):
                    pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches='tight')
                    plt.close()
        else:
            my_logger.info('calculate neighbors graph for following test ...')
            sc.pp.pca(A2R_predict)
            sc.pp.neighbors(A2R_predict)
            sc.pp.pca(R2A_predict)
            sc.pp.neighbors(R2A_predict)
                    
        """ test cluster index if needed """
        if test_cluster:
            index_R2A = calculate_cluster_index(R2A_predict)
            index_A2R = calculate_cluster_index(A2R_predict)
            
            index_matrix = pd.DataFrame([index_R2A, index_A2R])
            index_matrix.columns = ['ARI', 'AMI', 'NMI', 'HOM', 'COM']
            index_matrix.index = ['R2A', 'A2R']
            index_matrix.to_csv(output_path + '/cluster_index.csv')
            
        if test_evaluate:
            my_logger.info('evaluating correlation ...')
            pearson_list = []
            spearman_list = []
            R2A_predict_dense = R2A_predict.X.todense()
            ADT_data_train_dense = self.ATAC_data[test_id_a, :]
            for i in range(R2A_predict_dense.shape[0]):
                pearson_list.append(pearson(R2A_predict_dense[i].A[0], ADT_data_train_dense[i]))
                spearman_list.append(spearman(R2A_predict_dense[i].A[0], ADT_data_train_dense[i]))
            pearson_index = np.mean(pearson_list)
            spearma_index = np.mean(spearman_list)
            index_matrix = pd.DataFrame([[pearson_index, spearma_index]])
            index_matrix.columns = ['pearson', 'spearman']
            index_matrix.index = ['R2A']
            index_matrix.to_csv(output_path + '/relation.csv')

        """ save predicted model if needed """
        if output_data and not os.path.exists(output_path + '/predict'):
            my_logger.warning('trying to write predict to path: '+str(output_path)+'/predict')
            os.mkdir(output_path + '/predict')
            A2R_predict.write_h5ad(output_path + '/predict/A2R.h5ad')
            R2A_predict.write_h5ad(output_path + '/predict/R2A.h5ad')
            
        if return_predict:
            return A2R_predict, R2A_predict