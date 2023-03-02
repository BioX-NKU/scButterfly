#!/bin/bash

FILE_PATH="/home/atac2rna/data/atac2rna/model_output/butterfly/debug"
mkdir "$FILE_PATH"
for dataset in "cellline" "brain" "cellline" "chen" "kidney" "pbmc" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/five_fold_cross_validation/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done

FILE_PATH="/RNA_ATAC_output/five_fold_cross_validation/basic"
mkdir "$FILE_PATH"
for dataset in "bmmc" "brain" "cellline" "chen" "kidney" "pbmc" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/five_fold_cross_validation/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/five_fold_cross_validation/celltype_amp"
mkdir "$FILE_PATH"
for dataset in "bmmc" "brain" "cellline" "chen" "kidney" "pbmc" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/five_fold_cross_validation/run_model.py --model celltype_amplification --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/five_fold_cross_validation/multiVI_amp"
mkdir "$FILE_PATH"
for dataset in "bmmc" "brain" "cellline" "chen" "kidney" "pbmc" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/five_fold_cross_validation/run_model.py --model multiVI_amplification --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done