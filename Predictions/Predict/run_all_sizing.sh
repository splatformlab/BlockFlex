#!/bin/bash

#WLS=(terasort)
WLS=(terasort ml_prep pagerank)
IN=/home/breidys2/ml_res/inputs/sz
#WINDOWS=(60 120 180 240 300 360 420 480 540 600)
#TESTING BUFFERS, THIS IS PATH OF LEAST RESITSANCE
#WINDOWS=("1" "1.05" "1.1" "1.2" "1.3" "1.4")
WINDOWS=("1")
OUT=/home/breidys2/ml_res/outputs/sz
COPY=0
PRED=1
ACC=0
MISPRED=0
CYC=1

if [ $COPY -gt 0 ]
then
    #First we copy everything over
    echo "Copying files first"
    #for ((j=0;j<${#WINDOWS[@]};j++)); do
    for FILE in "${WLS[@]}"; do
        for i in {1..5}; do
            echo "Copying over ${FILE} ${i}"
            echo $IN
            cp "/home/breidys2/bench_parse/inputs/sz/${FILE}_${i}_sizing" $IN
        done
    done
    #done
fi

if [ $PRED -gt 0 ]
then
    echo "Running predictions"
    for ((j=0;j<${#WINDOWS[@]};j++)); do
        cnt=0
        for FILE in "${WLS[@]}"; do
            for i in {1..5}; do
                #Running each 5 times
                #for k in {1..5}; do
                    echo "Running ${FILE} ${i}"
                    cnt=$((cnt+1))
                    #python3 lstm_sz.py "${IN}/${FILE}_${j}_${i}_sizing" > "${OUT}/${FILE}_${j}_${i}.out" &
                    python3 lstm_sz.py "${IN}/${FILE}_${i}_sizing" $j > "${OUT}/${FILE}_${j}_${cnt}.out" 
                #done
            done
            wait
        done
    done
    for ((j=0;j<${#WINDOWS[@]};j++)); do
        cnt=0
        mult=0
        for file in ${IN}/*.txt; do
            #for i in {1..5}; do
                echo $file
                python3 lstm_sz_gl.py $file $j > "${OUT}/google_${j}_${mult}.out" &
                mult=$((mult+1))
            #done
            cnt=$((cnt+1))
            wait
            if [ $cnt == 5 ] 
            then
                break
            fi
        done
    done
fi

if [ $ACC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing accuracy output"
    python3 acc_parser.py sz sz_acc_out.txt 

    echo "Outputting Accuracy Results"
    cat outputs/sz_acc_out.txt
fi
if [ $MISPRED -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing mispred output"
    python3 mispred_parser.py sz sz_mispred_out.txt 

    echo "Outputting Mispred Results"
    cat outputs/sz_mispred_out.txt
fi
if [ $CYC -gt 0 ]
then
    #Now we parse and output some results
    echo "Parsing overhead output"
    python3 cyc_parser.py sz sz_cyc_out.txt 

    echo "Outputting Overhead Results"
    cat outputs/sz_cyc_out.txt
fi

#Done
echo "Done"
