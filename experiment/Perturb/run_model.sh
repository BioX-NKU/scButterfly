#!/bin/bash

FILE_PATH="/Perturb/basic"
mkdir "$FILE_PATH"
for dataset in "pbmc"
do
    mkdir "$FILE_PATH/$dataset"
    for number in "1" "2" "3" "4" "5" "6" "7"
    do
    {
        mkdir "$FILE_PATH/$dataset/$number"
        python /home/atac2rna/program/atac2rna/Model/butterfly/experiment/Perturb/run_model.py --model basic --data $dataset --file "$FILE_PATH/$dataset/$number" --number $number
    }
    done
done
