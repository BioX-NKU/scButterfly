import getopt
import sys
import gc
import os
from scButterfly.data_processing import RNA_data_preprocessing, ATAC_data_preprocessing
import scanpy as sc
import anndata as ad
from scButterfly.logger import *
from scButterfly.model_utlis import *
import warnings
import torch
import torch.nn as nn
from scButterfly.train_model import Model
import torch
import torch.nn as nn
warnings.filterwarnings("ignore")

class Butterfly():
    def __init__(
        self,
     ):
        """
        Butterfly model.

        """
        setup_seed(19193)
        self.my_logger = create_logger(name='Butterfly', ch=True, fh=False, levelname=logging.INFO, overwrite=False)
    
    def load_data(
        self,
        RNA_data,
        ATAC_data,
        train_id,
        test_id,
        validation_id = None,
     ):
        """
        Load data to Butterfly model.
        
        Parameters
        ----------
        train_id: list
            list of cell ids for training.
            
        test_id: list
            list of cell ids for testing.
            
        validation_id: list
            list of cell ids for validation, if None, Butterfly will use default 20% train cells for validation.

        """
        self.RNA_data = RNA_data.copy()
        self.ATAC_data = ATAC_data.copy()
        self.test_id_r = test_id.copy()
        self.test_id_a = test_id.copy()
        if validation_id is None:
            self.train_id = train_id
            random.shuffle(self.train_id)
            train_count = int(len(self.train_id)*0.8)
            self.train_id_r = self.train_id[0:train_count].copy()
            self.train_id_a = self.train_id[0:train_count].copy()
            self.validation_id_r = self.train_id[train_count:].copy()
            self.validation_id_a = self.train_id[train_count:].copy()
            del self.train_id
        else:
            self.train_id_r = train_id.copy()
            self.train_id_a = train_id.copy()
            self.validation_id_r = validation_id.copy()
            self.validation_id_a = validation_id.copy()
        self.my_logger.info('successfully load in data with\n\nRNA data:\n'+str(RNA_data)+'\n\nATAC data:\n'+str(ATAC_data))
        self.is_processed = False
        
    def data_preprocessing(
        self,
        normalize_total=True,
        log1p=True,
        use_hvg=True,
        n_top_genes=3000,
        binary_data=True,
        filter_features=True,
        fpeaks=0.005,
        tfidf=True,
        normalize=True,
        save_data=False,
        file_path=None,
        logging_path=None
    ):
        """
        Preprocessing for RNA data and ATAC data in Butterfly.

        Parameters
        ----------
        normalize_total: bool
            choose use normalization or not, default True.

        log1p: bool
            choose use log transformation or not, default True.

        use_hvg: bool
            choose use highly variable genes or not, default True.

        n_top_genes: int
            the count of highly variable genes, if not use highly variable, set use_hvg = False and n_top_genes = None, default 3000.
            
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
            
         """
        if self.is_processed:
            self.my_logger.warning('finding data have already been processed!')
        else:
            self.RNA_data_p = RNA_data_preprocessing(
                self.RNA_data,
                normalize_total=normalize_total,
                log1p=log1p,
                use_hvg=use_hvg,
                n_top_genes=n_top_genes,
                save_data=save_data,
                file_path=file_path,
                logging_path=file_path
                )
            self.ATAC_data_p = ATAC_data_preprocessing(
                self.ATAC_data,
                binary_data=binary_data,
                filter_features=filter_features,
                fpeaks=fpeaks,
                tfidf=tfidf,
                normalize=normalize,
                save_data=save_data,
                file_path=file_path,
                logging_path=file_path
            )[0]
            self.is_processed = True
            
    def amplification(
        self,
        amp_type=None,
        MultiVI_path=None
    ):
        """
        Data amplification for Butterfly model.
        
        Parameters
        ----------
        amp_type: str
            Butterfly support two types of amp_type, "cell_type_amplification" and "MultiVI_amplification". "cell_type_amplification" need "cell_type" in RNA_data.obs and ATAC_data.obs， default None.
            
        MultiVI_path: str
            path for pretrained MultiVI model, if None, Butterfly will train a MultiVI model first, default None.

        """
        if amp_type == 'cell_type_amplification' and 'cell_type' in self.RNA_data.obs.keys():
            self.my_logger.info('using data amplification with cell type labels.')
            copy_count = 3
            random.seed(19193)
            self.ATAC_data.obs.index = [str(i) for i in range(len(self.ATAC_data.obs.index))]
            cell_type = self.ATAC_data.obs.cell_type.iloc[self.train_id_a]
            for i in range(len(cell_type.cat.categories)):
                cell_type_name = cell_type.cat.categories[i]
                idx_temp = list(cell_type[cell_type == cell_type_name].index.astype(int))
                for j in range(copy_count - 1):
                    random.shuffle(idx_temp)
                    self.train_id_r.extend(idx_temp)
                    random.shuffle(idx_temp)
                    self.train_id_a.extend(idx_temp)
        if amp_type == 'cell_type_amplification' and not 'cell_type' in self.RNA_data.obs.keys():
            self.my_logger.warning('not find "cell_type" in data.obs, trying to use MultiVI amplification ...')
            amp_type = 'MultiVI_amplification'
        if amp_type == 'MultiVI_amplification':
            self.my_logger.info('using data amplification with MultiVI cluster labels.')
            from scvi_colab import install
            import pandas as pd
            install()
            import scvi
            import sys
            import scipy.sparse as sp

            adata = ad.AnnData(sp.hstack((self.RNA_data.X, self.ATAC_data.X)))
            adata.X = adata.X.tocsr()
            adata.obs = self.RNA_data.obs

            m = len(self.RNA_data.var.index)
            n = len(self.ATAC_data.var.index)
            adata.var.index = pd.Series([self.RNA_data.var.index[i] if i<m else self.ATAC_data.var.index[i-m] for i in range(m+n)], dtype='object')
            adata.var['modality'] = pd.Series(['Gene Expression' if i<m else 'Peaks' for i in range(m+n)], dtype='object').values

            adata.var_names_make_unique()

            adata_mvi = scvi.data.organize_multiome_anndatas(adata)

            adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

            sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))

            train_id, validation_id, test_id = self.train_id_r, self.validation_id_r, self.test_id_r

            train_adata = idx2adata_multiVI(adata_mvi, train_id, validation_id, test_id)[0]
            
            if MultiVI_path is None:
                self.my_logger.info('no trained model find, train MultiVI model first ...')
                
                scvi.model.MULTIVI.setup_anndata(train_adata, batch_key='modality')

                mvi = scvi.model.MULTIVI(
                    train_adata,
                    n_genes=(adata_mvi.var['modality']=='Gene Expression').sum(),
                    n_regions=(adata_mvi.var['modality']=='Peaks').sum(),
                )

                mvi.train()
            else:
                self.my_logger.info('load trained MultiVI model from '+MultiVI_path)
                mvi = scvi.model.MULTIVI.load(MultiVI_path, adata=train_adata)

            train_adata.obsm["MultiVI_latent"] = mvi.get_latent_representation()
            leiden_adata = ad.AnnData(train_adata.obsm["MultiVI_latent"])
            sc.pp.neighbors(leiden_adata)
            sc.tl.leiden(leiden_adata, resolution=3)

            del train_adata, adata_mvi
            gc.collect()
            
            
            copy_count = 3
            random.seed(19193)
            cell_type = leiden_adata.obs.leiden
            for i in range(len(cell_type.cat.categories)):
                cell_type_name = cell_type.cat.categories[i]
                idx_temp = list(cell_type[cell_type == cell_type_name].index.astype(int))
                for j in range(copy_count - 1):
                    random.shuffle(idx_temp)
                    for each in idx_temp:
                        self.train_id_r.append(train_id[each])
                    random.shuffle(idx_temp)
                    for each in idx_temp:
                        self.train_id_a.append(train_id[each])

    def construct_model(
        self,
        chrom_list,
        logging_path = None,
        R_encoder_nlayer = 2, 
        A_encoder_nlayer = 2,
        R_decoder_nlayer = 2, 
        A_decoder_nlayer = 2,
        R_encoder_dim_list = [256, 128],
        A_encoder_dim_list = [32, 128],
        R_decoder_dim_list = [128, 256],
        A_decoder_dim_list = [128, 32],
        R_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_encoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        R_decoder_act_list = [nn.LeakyReLU(), nn.LeakyReLU()],
        A_decoder_act_list = [nn.LeakyReLU(), nn.Sigmoid()],
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
        A_noise_rate = 0.3,
    ):
        """
        Main model.
        
        Parameters
        ----------
        chrom_list: list
            list of peaks count for each chromosomes.
            
        logging_path: str
            the path for output process logging, if not save, set it None, default None.

        R_encoder_nlayer: int
            layer counts of RNA encoder, default 2.
            
        A_encoder_nlayer: int
            layer counts of ATAC encoder, default 2.
            
        R_decoder_nlayer: int
            layer counts of RNA decoder, default 2.
            
        A_decoder_nlayer: int
            layer counts of ATAC decoder, default 2.
            
        R_encoder_dim_list: list
            dimension list of RNA encoder, length equal to R_encoder_nlayer, default [256, 128].
            
        A_encoder_dim_list: list
            dimension list of ATAC encoder, length equal to A_encoder_nlayer, default [32, 128].
            
        R_decoder_dim_list: list
            dimension list of RNA decoder, length equal to R_decoder_nlayer, default [128, 256].
            
        A_decoder_dim_list: list
            dimension list of ATAC decoder, length equal to A_decoder_nlayer, default [128, 32].
            
        R_encoder_act_list: list
            activation list of RNA encoder, length equal to R_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        A_encoder_act_list: list
            activation list of ATAC encoder, length equal to A_encoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        R_decoder_act_list: list
            activation list of RNA decoder, length equal to R_decoder_nlayer, default [nn.LeakyReLU(), nn.LeakyReLU()].
            
        A_decoder_act_list: list
            activation list of ATAC decoder, length equal to A_decoder_nlayer, default [nn.LeakyReLU(), nn.Sigmoid()].
            
        translator_embed_dim: int
            dimension of embedding space for translator, default 128.
            
        translator_input_dim_r: int
            dimension of input from RNA encoder for translator, default 128.
            
        translator_input_dim_a: int
            dimension of input from ATAC encoder for translator, default 128.
            
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
            rate of set part of ATAC input data to 0, default 0.3.

        """
        R_encoder_dim_list.insert(0, self.RNA_data_p.X.shape[1]) 
        A_encoder_dim_list.insert(0, self.ATAC_data_p.X.shape[1])
        R_decoder_dim_list.append(self.RNA_data_p.X.shape[1]) 
        A_decoder_dim_list.append(self.ATAC_data_p.X.shape[1])
        A_encoder_dim_list[1] *= len(chrom_list)
        A_decoder_dim_list[1] *= len(chrom_list)
        self.model = Model(
            R_encoder_nlayer = R_encoder_nlayer, 
            A_encoder_nlayer = A_encoder_nlayer,
            R_decoder_nlayer = R_decoder_nlayer, 
            A_decoder_nlayer = A_decoder_nlayer,
            R_encoder_dim_list = R_encoder_dim_list,
            A_encoder_dim_list = A_encoder_dim_list,
            R_decoder_dim_list = R_decoder_dim_list,
            A_decoder_dim_list = A_decoder_dim_list,
            R_encoder_act_list = R_encoder_act_list,
            A_encoder_act_list = A_encoder_act_list,
            R_decoder_act_list = R_decoder_act_list,
            A_decoder_act_list = A_decoder_act_list,
            translator_embed_dim = translator_embed_dim, 
            translator_input_dim_r = translator_input_dim_r,
            translator_input_dim_a = translator_input_dim_a,
            translator_embed_act_list = translator_embed_act_list,
            discriminator_nlayer = discriminator_nlayer,
            discriminator_dim_list_R = discriminator_dim_list_R,
            discriminator_dim_list_A = discriminator_dim_list_A,
            discriminator_act_list = discriminator_act_list,
            dropout_rate = dropout_rate,
            R_noise_rate = R_noise_rate,
            A_noise_rate = A_noise_rate,
            chrom_list = chrom_list,
            logging_path = logging_path,
            RNA_data = self.RNA_data_p,
            ATAC_data = self.ATAC_data_p
        )
        self.my_logger.info('successfully construct butterfly model.')

    def train_model(
        self,
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
        a_loss = nn.BCELoss(size_average=True),
        d_loss = nn.BCELoss(size_average=True),
        loss_weight = [1, 2, 1],
        output_path = None,
        seed = 19193,
        kl_mean = True,
        R_pretrain_kl_warmup = 50,
        A_pretrain_kl_warmup = 50,
        translation_kl_warmup = 50,
        load_model = None,
        logging_path = None
    ):
        """
        Training for model.
        
        Parameters
        ----------
        R_encoder_lr: float
            learning rate of RNA encoder, default 0.001.
            
        A_encoder_lr: float
            learning rate of ATAC encoder, default 0.001.
            
        R_decoder_lr: float
            learning rate of RNA decoder, default 0.001.
            
        A_decoder_lr: float
            learning rate of ATAC decoder, default 0.001.
       
        R_translator_lr: float
            learning rate of RNA pretrain translator, default 0.001.
            
        A_translator_lr: float
            learning rate of ATAC pretrain translator, default 0.001.
            
        translator_lr: float
            learning rate of translator, default 0.001.
            
        discriminator_lr: float
            learning rate of discriminator, default 0.005.
            
        R2R_pretrain_epoch: int
            max epoch for pretrain RNA autoencoder, default 100.
            
        A2A_pretrain_epoch: int
            max epoch for pretrain ATAC autoencoder, default 100.
            
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
            loss function for ATAC reconstruction, default nn.BCELoss(size_average=True).
            
        d_loss
            loss function for discriminator, default nn.BCELoss(size_average=True).
            
        loss_weight: list
            list of loss weight for [r_loss, a_loss, d_loss], default [1, 2, 1].

        output_path: str
            file path for model output, default None.
            
        seed: int
            set up the random seed, default 19193.
            
        kl_mean: bool
            size average for kl divergence or not, default True.
            
        R_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in RNA pretrain, default 50.
        
        A_pretrain_kl_warmup: int
            epoch of linear weight warm up for kl divergence in ATAC pretrain, default 50.
        
        translation_kl_warmup: int
            epoch of linear weight warm up for kl divergence in translator pretrain, default 50.
            
        load_model: str
            the path for loading model if needed, else set it None, default None.
            
        logging_path: str
            the path for output process logging, if not save, set it None, default None.
            
        """
        self.my_logger.info('training butterfly model ...')
        R_kl_div = 1 / self.RNA_data_p.X.shape[1] * 20
        A_kl_div = 1 / self.ATAC_data_p.X.shape[1] * 20
        kl_div = R_kl_div + A_kl_div
        loss_weight.extend([R_kl_div, A_kl_div, kl_div])
        self.model.train(
            R_encoder_lr = R_encoder_lr,
            A_encoder_lr = A_encoder_lr,
            R_decoder_lr = R_decoder_lr,
            A_decoder_lr = A_decoder_lr,
            R_translator_lr = R_translator_lr,
            A_translator_lr = A_translator_lr,
            translator_lr = translator_lr,
            discriminator_lr = discriminator_lr,
            R2R_pretrain_epoch = R2R_pretrain_epoch,
            A2A_pretrain_epoch = A2A_pretrain_epoch,
            lock_encoder_and_decoder = lock_encoder_and_decoder,
            translator_epoch = translator_epoch,
            patience = patience,
            batch_size = batch_size,
            r_loss = r_loss,
            a_loss = a_loss,
            d_loss = d_loss,
            loss_weight = loss_weight,
            train_id_r = self.train_id_r,
            train_id_a = self.train_id_a,
            validation_id_r = self.validation_id_r, 
            validation_id_a = self.validation_id_a, 
            output_path = output_path,
            seed = seed,
            kl_mean = kl_mean,
            R_pretrain_kl_warmup = R_pretrain_kl_warmup,
            A_pretrain_kl_warmup = A_pretrain_kl_warmup,
            translation_kl_warmup = translation_kl_warmup,
            load_model = load_model,
            logging_path = logging_path
        )

    def test_model(
        self, 
        model_path = None,
        load_model = False,
        output_path = None,
        test_relation = False,
        test_cluster = False,
        test_figure = False,
        output_data = False
    ):
        """
        Test for model.
        
        Parameters
        ----------
        model_path: str
            path for load trained model, default None.
            
        load_model: bool
            load the pretrained model or not, default False.
            
        output_path: str
            file path for model output, default None.
            
        test_relation: bool
            test correlation or not, default False.
            
        test_cluster: bool
            test clustrer index or not, default False.
            
        test_figure: bool
            test tSNE or not, default False.
            
        output_data: bool
            output the predicted test data to file or not, default False.
            
        """
        self.my_logger.info('testing butterfly model ...')
        A2R_predict, R2A_predict = self.model.test(
            test_id_r = self.test_id_r,
            test_id_a = self.test_id_a, 
            model_path = model_path,
            load_model = load_model,
            output_path = output_path,
            test_evaluate = test_relation,
            test_cluster = test_cluster,
            test_figure = test_figure,
            output_data = output_data,
            return_predict = True
        )
        return A2R_predict, R2A_predict