#!/bin/bash

FILE_PATH="/RNA_ATAC_output/unpair_data/basic"
mkdir "$FILE_PATH"
for dataset in "muto"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/unpair_data/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/unpair_data/scglue_integ"
mkdir "$FILE_PATH"
for dataset in "muto"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/unpair_data/run_model.py --model scglue_integ --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/unpair_data/basic"
mkdir "$FILE_PATH"
for dataset in "yao"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/unpair_data/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
FILE_PATH="/RNA_ATAC_output/unpair_data/scglue_integ"
mkdir "$FILE_PATH"
for dataset in "yao"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/RNA_ATAC/unpair_data/run_model.py --model scglue_integ --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done