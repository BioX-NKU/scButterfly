import torch
import torch.nn as nn
import torch.nn.functional as F


class NetBlock(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list, 
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        multiple layers netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
        
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(NetBlock, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            
            self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            nn.init.xavier_uniform_(self.linear_list[i].weight)
            self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
            self.activation_list.append(act_list[i])
            if not i == nlayer -1: 
                self.dropout_list.append(nn.Dropout(dropout_rate))
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            x = self.linear_list[i](x)
            x = self.bn_list[i](x)
            x = self.activation_list[i](x)
            if not i == self.nlayer -1:
                """ don't use dropout for output to avoid loss calculate break down """
                x = self.dropout_list[i](x)

        return x

    
class Split_Chrom_Encoder_block(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list,
        chrom_list: list,
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        ATAC encoder netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
            
        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[0].
            
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(Split_Chrom_Encoder_block, self).__init__()
        self.nlayer = nlayer
        self.chrom_list = chrom_list
        self.noise_dropout = nn.Dropout(noise_rate)
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            if i == 0:
                """first layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(chrom_list[j], dim_list[i + 1] // len(chrom_list)))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(dim_list[i + 1] // len(chrom_list)))
                    self.activation_list[i].append(act_list[i])
                    self.dropout_list[i].append(nn.Dropout(dropout_rate))
            else:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                if not i == nlayer -1: 
                    self.dropout_list.append(nn.Dropout(dropout_rate))
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if i == 0:
                x = torch.split(x, self.chrom_list, dim = 1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.dropout_list[0][j](self.activation_list[0][j](self.bn_list[0][j](self.linear_list[0][j](x[j])))))
                x = torch.concat(temp, dim = 1)
            else:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                if not i == self.nlayer -1:
                    """ don't use dropout for output to avoid loss calculate break down """
                    x = self.dropout_list[i](x)
        return x


class Split_Chrom_Decoder_block(nn.Module):
    def __init__(
        self,
        nlayer: int, 
        dim_list: list, 
        act_list: list,
        chrom_list: list,
        dropout_rate: float, 
        noise_rate: float
        ):
        """
        ATAC decoder netblock with specific layer counts, dimension, activations and dropout.
        
        Parameters
        ----------
        nlayer
            layer counts.
            
        dim_list
            dimension list, length equal to nlayer + 1.
        
        act_list
            activation list, length equal to nlayer + 1.
            
        chrom_list
            list record the peaks count for each chrom, assert that sum of chrom list equal to dim_list[end].
            
        dropout_rate
            rate of dropout.
        
        noise_rate
            rate of set part of input data to 0.
            
        """
        super(Split_Chrom_Decoder_block, self).__init__()
        self.nlayer = nlayer
        self.noise_dropout = nn.Dropout(noise_rate)
        self.chrom_list = chrom_list
        self.linear_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.activation_list = nn.ModuleList()
        self.dropout_list = nn.ModuleList()
        
        for i in range(nlayer):
            if not i == nlayer -1:
                self.linear_list.append(nn.Linear(dim_list[i], dim_list[i + 1]))
                nn.init.xavier_uniform_(self.linear_list[i].weight)
                self.bn_list.append(nn.BatchNorm1d(dim_list[i + 1]))
                self.activation_list.append(act_list[i])
                self.dropout_list.append(nn.Dropout(dropout_rate))
            else:
                """last layer seperately forward for each chrom"""
                self.linear_list.append(nn.ModuleList())
                self.bn_list.append(nn.ModuleList())
                self.activation_list.append(nn.ModuleList())
                self.dropout_list.append(nn.ModuleList())
                for j in range(len(chrom_list)):
                    self.linear_list[i].append(nn.Linear(dim_list[i] // len(chrom_list), chrom_list[j]))
                    nn.init.xavier_uniform_(self.linear_list[i][j].weight)
                    self.bn_list[i].append(nn.BatchNorm1d(chrom_list[j]))
                    self.activation_list[i].append(act_list[i])
        
    
    def forward(self, x):

        x = self.noise_dropout(x)
        for i in range(self.nlayer):
            if not i == self.nlayer -1:
                x = self.linear_list[i](x)
                x = self.bn_list[i](x)
                x = self.activation_list[i](x)
                x = self.dropout_list[i](x)
            else:
                x = torch.chunk(x, len(self.chrom_list), dim = 1)
                temp = []
                for j in range(len(self.chrom_list)):
                    temp.append(self.activation_list[i][j](self.bn_list[i][j](self.linear_list[i][j](x[j]))))
                x = torch.concat(temp, dim = 1)

        return x
    
    
class Translator(nn.Module):
    def __init__(
        self,
        translator_input_dim_r: int, 
        translator_input_dim_a: int, 
        translator_embed_dim: int, 
        translator_embed_act_list: list, 
        ):
        """
        Translator block used for translation between different omics.
        
        Parameters
        ----------
        translator_input_dim_r
            dimension of input from RNA encoder for translator.
            
        translator_input_dim_a
            dimension of input from ATAC encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        """
        
        super(Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.RNA_encoder_l_mu = nn.Linear(translator_input_dim_r, translator_embed_dim)
        nn.init.xavier_uniform_(self.RNA_encoder_l_mu.weight)
        self.RNA_encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.RNA_encoder_act_mu = mean_activation
        
        self.ATAC_encoder_l_mu = nn.Linear(translator_input_dim_a, translator_embed_dim)
        nn.init.xavier_uniform_(self.ATAC_encoder_l_mu.weight)
        self.ATAC_encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.ATAC_encoder_act_mu = mean_activation
        
        self.RNA_encoder_l_d = nn.Linear(translator_input_dim_r, translator_embed_dim)
        nn.init.xavier_uniform_(self.RNA_encoder_l_d.weight)
        self.RNA_encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.RNA_encoder_act_d = log_var_activation
        
        self.ATAC_encoder_l_d = nn.Linear(translator_input_dim_a, translator_embed_dim)
        nn.init.xavier_uniform_(self.ATAC_encoder_l_d.weight)
        self.ATAC_encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.ATAC_encoder_act_d = log_var_activation
        
        self.RNA_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_r)
        nn.init.xavier_uniform_(self.RNA_decoder_l.weight)
        self.RNA_decoder_bn = nn.BatchNorm1d(translator_input_dim_r)
        self.RNA_decoder_act = decoder_activation
        
        self.ATAC_decoder_l = nn.Linear(translator_embed_dim, translator_input_dim_a)
        nn.init.xavier_uniform_(self.ATAC_decoder_l.weight)
        self.ATAC_decoder_bn = nn.BatchNorm1d(translator_input_dim_a)
        self.ATAC_decoder_act = decoder_activation
                
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward_with_RNA(self, x, forward_type):
            
        latent_layer_mu = self.RNA_encoder_act_mu(self.RNA_encoder_bn_mu(self.RNA_encoder_l_mu(x)))
        latent_layer_d = self.RNA_encoder_act_d(self.RNA_encoder_bn_d(self.RNA_encoder_l_d(x)))
        
        if forward_type == 'test':
            latent_layer = latent_layer_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(latent_layer_mu, latent_layer_d)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_A_out = self.ATAC_decoder_act(self.ATAC_decoder_bn(self.ATAC_decoder_l(latent_layer)))

        return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d
    
    def forward_with_ATAC(self, x, forward_type):
        # the function with ATAC input and RNA, ATAC output 
            
        latent_layer_mu = self.ATAC_encoder_act_mu(self.ATAC_encoder_bn_mu(self.ATAC_encoder_l_mu(x)))
        latent_layer_d = self.ATAC_encoder_act_d(self.ATAC_encoder_bn_d(self.ATAC_encoder_l_d(x)))
        
        if forward_type == 'test':
            latent_layer = latent_layer_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(latent_layer_mu, latent_layer_d)
        
        latent_layer_R_out = self.RNA_decoder_act(self.RNA_decoder_bn(self.RNA_decoder_l(latent_layer)))
        latent_layer_A_out = self.ATAC_decoder_act(self.ATAC_decoder_bn(self.ATAC_decoder_l(latent_layer)))

        return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d
    
    def train_model(self, x, input_type):
        if input_type == 'RNA':
            latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d = self.forward_with_RNA(x, 'train')
            return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d
        elif input_type == 'ATAC':
            latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d = self.forward_with_ATAC(x, 'train')
            return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d
        
    def test_model(self, x, input_type):
        if input_type == 'RNA':
            latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d = self.forward_with_RNA(x, 'test')
            return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d
        elif input_type == 'ATAC':
            latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d = self.forward_with_ATAC(x, 'test')
            return latent_layer_R_out, latent_layer_A_out, latent_layer_mu, latent_layer_d

        
class Single_Translator(nn.Module):
    def __init__(
        self,
        translator_input_dim: int,  
        translator_embed_dim: int, 
        translator_embed_act_list: list, 
        ):
        """
        Single translator block used only for pretraining.
        
        Parameters
        ----------
        translator_input_dim
            dimension of input from encoder for translator.
            
        translator_embed_dim
            dimension of embedding space for translator.
            
        translator_embed_act_list
            activation list for translator, involving [mean_activation, log_var_activation, decoder_activation].
            
        """
        
        super(Single_Translator, self).__init__()
        
        mean_activation, log_var_activation, decoder_activation = translator_embed_act_list
        
        self.encoder_l_mu = nn.Linear(translator_input_dim, translator_embed_dim)
        nn.init.xavier_uniform_(self.encoder_l_mu.weight)
        self.encoder_bn_mu = nn.BatchNorm1d(translator_embed_dim)
        self.encoder_act_mu = mean_activation
        
        self.encoder_l_d = nn.Linear(translator_input_dim, translator_embed_dim)
        nn.init.xavier_uniform_(self.encoder_l_d.weight)
        self.encoder_bn_d = nn.BatchNorm1d(translator_embed_dim)
        self.encoder_act_d = log_var_activation

        self.decoder_l = nn.Linear(translator_embed_dim, translator_input_dim)
        nn.init.xavier_uniform_(self.decoder_l.weight)
        self.decoder_bn = nn.BatchNorm1d(translator_input_dim)
        self.decoder_act = decoder_activation                
    
    def reparameterize(self, mu, sigma):
        sigma = torch.exp(sigma/2)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, x, forward_type):
            
        latent_layer_mu = self.encoder_act_mu(self.encoder_bn_mu(self.encoder_l_mu(x)))
        latent_layer_d = self.encoder_act_d(self.encoder_bn_d(self.encoder_l_d(x)))
        
        if forward_type == 'test':
            latent_layer = latent_layer_mu
        elif forward_type == 'train':
            latent_layer = self.reparameterize(latent_layer_mu, latent_layer_d)
        
        latent_layer_out = self.decoder_act(self.decoder_bn(self.decoder_l(latent_layer)))
        return latent_layer_out, latent_layer_mu, latent_layer_d
