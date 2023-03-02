#!/bin/bash

FILE_PATH="/RNA_ADT_output/five_fold_cross_validation/basic"
mkdir "$FILE_PATH"
for dataset in "seurat" "bmmc_adt"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ADT/five_fold_cross_validation/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ADT_output/five_fold_cross_validation/celltype_amp"
mkdir "$FILE_PATH"
for dataset in "seurat" "bmmc_adt"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ADT/five_fold_cross_validation/run_model.py --model celltype_amplification --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ADT_output/five_fold_cross_validation/totalVI_amp"
mkdir "$FILE_PATH"
for dataset in "seurat" "bmmc_adt"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ADT/five_fold_cross_validation/run_model.py --model totalVI_amplification --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
