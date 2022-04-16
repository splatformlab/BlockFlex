#!/bin/bash

WLS=(terasort ml_prep pagerank)
IN=/home/breidys2/ml_res/inputs/dur_sz
#WINDOWS=(60 120 180 240 300 360 420 480 540 600)
#TESTING BUFFERS WITH THIS
#WINDOWS=("1" "1.05" "1.1" "1.2" "1.5" "2")
WINDOWS=("1")

OUT=/home/breidys2/ml_res/outputs/dur_sz
COPY=0
PRED=1
ACC=0
MISPRED=0
CYC=1

if [ $COPY -gt 0 ]
then
    #First we copy everything over
    echo "Copying files first"
    #for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
    for FILE in "${WLS[@]}"; do
        for i in {1..5}; do
            echo "Copying over ${FILE} ${j} ${i}"
            echo $IN
            cp "/home/breidys2/bench_parse/inputs/dur_sz/${FILE}_${i}_dur" $IN
        done
    done
    #done
fi

if [ $PRED -gt 0 ]
then
    echo "Running predictions"
    #Now we use this
    for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                echo "Running ${FILE} ${i}"
                #python3 lstm_dur_bw.py "${IN}/${FILE}_${j}_${i}_dur" > "${OUT}/${FILE}_${j}_${i}.out" &
                python3 lstm_dur_bw.py "${IN}/${FILE}_${i}_dur" $j > "${OUT}/${FILE}_${j}_${i}.out" 
            done
            wait
        done
    done
    for (( j=0; j < ${#WINDOWS[@]}; j++ )); do
        cnt=0
        mult=0
        for file in ${IN}/*.txt; do
            #for i in {1..5}; do
                echo $file
                echo $cnt
                python3 lstm_dur_gl.py $file $j > "${OUT}/google_${j}_${mult}.out" 
                mult=$((mult+1))
            #done
            wait
            cnt=$((cnt+1))
            if [ $cnt == 5 ] 
            then
                break
            fi
        done
        wait
    done
fi

if [ $ACC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing accuracy output"
    python3 acc_parser.py dur_sz dur_sz_acc_out.txt 

    echo "Outputting Accuracy Results"
    cat outputs/dur_sz_acc_out.txt
fi
if [ $MISPRED -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing mispred output"
    python3 mispred_parser.py dur_sz dur_sz_mispred_out.txt 

    echo "Outputting Mispred Results"
    cat outputs/dur_sz_mispred_out.txt
fi
if [ $CYC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing overhead output"
    python3 cyc_parser.py dur_sz dur_sz_cyc_out.txt 

    echo "Outputting Overhead Results"
    cat outputs/dur_sz_cyc_out.txt
fi

#Done
echo "Done"
