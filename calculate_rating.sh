#!/bin/bash

rm -f models/rating.txt
#for data_set in "1_r" "2_r" "3_r" "4_c" "5_c" "6_c" "7_c" "8_c"
for data_set in "1_r"
do
    if [[ ${data_set: -1} = 'c' ]]
    then
        mode="classification"
    else
        mode="regression"
    fi
    printf $data_set >> models/rating.txt
    printf ":" >> models/rating.txt
    app/entrypoint.sh $mode $data_set
done

cat models/rating.txt
