#!/bin/bash

FILE_PATH="/RNA_ATAC_output/batch_generalization/basic"
mkdir "$FILE_PATH"
for dataset in "bmmc" "cellline" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/batch_generalization/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/batch_generalization/celltype_amp"
mkdir "$FILE_PATH"
for dataset in "bmmc" "cellline" "ma"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/batch_generalization/run_model.py --model celltype_amplification --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
