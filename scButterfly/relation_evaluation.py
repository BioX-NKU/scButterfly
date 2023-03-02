import imp
from statistics import mean
from turtle import forward
from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import spearmanr, pearsonr
import sys
from scipy import integrate

from urllib3 import Retry


#binrize ATAC
def binirize(atac_seq, every_mean):
    bin = []
    
    for i in range (len(atac_seq)):
        if atac_seq[i] < every_mean[i] :
            bin.append(0)
            #bin.append(atac_seq[i])
            
        else:
            bin.append(1)
            #bin.append(atac_seq[i])
    bin = np.array(bin)
    return bin

#PP expression rate
def PP (atac_seq):
    sum = 0
    for i in range(len(atac_seq)):
        if atac_seq[i]>0:
            sum += 1
    return sum/len(atac_seq)

#AUROC: RNA to ATAC
def AUROC(atac_seq_pre,atac_seq, every_mean):
    auroc = 0
    atac_seq = binirize(atac_seq,every_mean)

    try:
        auroc = roc_auc_score(atac_seq, atac_seq_pre)
    except ValueError:
        pass

    #auroc = roc_auc_score(atac_seq,atac_seq_pre)
    return auroc
def draw_AUROC(atac_seq_pre,atac_seq, every_mean,output_path):
    auroc = 0
    atac_seq = binirize(atac_seq,every_mean)
    fpr, tpr, thresholds = roc_curve(atac_seq, atac_seq_pre)
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.title("ROC")
    pyplot.xlabel("FPR")
    pyplot.ylabel("TPR")
    pyplot.savefig(output_path+'AUROC.pdf')
    #auroc = roc_auc_score(atac_seq,atac_seq_pre)
    return auroc

#AUPRC: RNA to ATAC
def AUPR_norm (atac_seq_pre,atac_seq, every_mean):
    atac_seq = binirize(atac_seq, every_mean)
    norm = 0  
    presion, recall, thee = precision_recall_curve(atac_seq,atac_seq_pre)
    sorted_index = np.argsort(presion)
    fpr_list_sorted =  np.array(presion)[sorted_index]
    tpr_list_sorted = np.array(recall)[sorted_index]
    integrate.trapz(y=tpr_list_sorted, x=fpr_list_sorted)
    aup = auc(fpr_list_sorted, tpr_list_sorted)
    pp = PP(atac_seq)
    norm = (aup-pp)/(1-pp)
    return aup

def draw_AUPR_norm (atac_seq_pre,atac_seq, every_mean,output_path):
    atac_seq = binirize(atac_seq, every_mean)
    norm = 0  
    presion, recall, thee = precision_recall_curve(atac_seq,atac_seq_pre)
    pyplot.plot(reacall, precision, 'r')
    pyplot.legend(loc='lower left')
    pyplot.xlabel('Recall')  # the name of x-axis
    pyplot.ylabel('Precision')  # the name of y-axis
    pyplot.title('Precision-Recall')  # the title of figure
    pyplot.savefig(output_path+'presicion_recall.pdf')
    return aup
#Pearson ATAC to RNA
def pearson (rna_seq_pre,rna_seq):
    person = pearsonr(rna_seq,rna_seq_pre)[0]
    return person

#Spearman ATAC to RNA
def spearman (rna_seq_pre,rna_seq):
    spe = spearmanr(rna_seq,rna_seq_pre)[0]
    return spe

#FOSCTTM 
def foscttm (atac_seq_pre,total_samples):
    fracs=calc_domainAveraged_FOSCTTM(total_samples, atac_seq_pre)
    fracs = np.mean(fracs)
    return fracs

#Only need this function to test a model
def evaluate(My_model, test_dataloader, batch_size, RNA_input_dim, ATAC_input_dim,loss_func, every_mean):
    
    mean_test_loss = 0
    mean_test_auroc = 0
    mean_test_aupr = 0
    mean_test_pearson = 0
    mean_test_spearman = 0
    sum_auroc = 0
    sum_aupr = 0
    max_enum = 9
    for idx, batch_samples in enumerate(test_dataloader):
        if torch.cuda.is_available():
            batch_samples = batch_samples.cuda()
        

        RNA_input, ATAC_input = torch.split(batch_samples, [RNA_input_dim, ATAC_input_dim], dim=1)
        R2R, R2A = My_model(RNA_input,'RNA')
        A2R, A2A = My_model(ATAC_input,'ATAC')
        R2, A2 = torch.cat([R2R, R2A], dim=1), torch.cat([A2R, A2A], dim=1)
        lossR_temp = loss_func(R2, batch_samples)
        lossA_temp = loss_func(A2, batch_samples)
        loss_temp = lossA_temp + lossR_temp


        atac_input = ATAC_input.cpu().numpy()
        rna_input = RNA_input.cpu().numpy()
        r2a = R2A.cpu().detach().numpy()
        a2r = A2R.cpu().detach().numpy()
        ss = sum(atac_input)

        #对每一个cell， 计算auroc和aupr，再取平均
        for i in range(len(atac_input)):
            mean_test_loss += loss_temp.item()
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

            finish = idx/len(test_dataloader)
            print(finish,"has been finished.")

        

    mean_test_loss /= len(test_dataloader)*batch_size
    mean_test_auroc /= sum_auroc
    mean_test_aupr /= sum_aupr
    mean_test_pearson /= len(test_dataloader)*batch_size
    mean_test_spearman /= len(test_dataloader)*batch_size
    print('*****mean_test_loss: ' + str(mean_test_loss) + '*****')
    print('*****mean_test_auroc: ' + str(mean_test_auroc) + '*****')
    print('*****mean_test_aupr: ' + str(mean_test_aupr) + '*****')
    print('*****mean_test_pearson: ' + str(mean_test_pearson) + '*****')
    print('*****mean_test_spearman: ' + str(mean_test_spearman) + '*****')
    return [mean_test_loss,mean_test_auroc,mean_test_aupr,mean_test_pearson,mean_test_spearman]

