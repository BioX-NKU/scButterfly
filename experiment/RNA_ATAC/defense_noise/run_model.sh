#!/bin/bash

FILE_PATH="/RNA_ATAC_output/defense_noise/basic"
mkdir "$FILE_PATH"
for noise_fold in "1" "2" "3" "4" "5"
do
    mkdir "$FILE_PATH/$noise_fold"
    for dataset in "chen"
    do
        mkdir "$FILE_PATH/$noise_fold/$dataset"
        for number in "1" "2" "3" "4" "5" "6" "7" "8" "9"
        do
        {
            mkdir "$FILE_PATH/$noise_fold/$dataset/$number"
           python /program/atac2rna/Model/butterfly/experiment/RNA_ATAC/defense_noise/run_model.py --model basic --data $dataset --file "$FILE_PATH/$noise_fold/$dataset/$number" --number $number --noise_fold $noise_fold
        }
        done

    done
done