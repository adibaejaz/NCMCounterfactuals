#!/bin/bash

for gpu_id in 0 1 4 5 6 7
do
    for i in {1..5}
    do
        session_name="gpu_${gpu_id}_run_${i}"
        sleep 20
        tmux new-session -d -s $session_name "python score_experiments.py --gpu $gpu_id; exec bash"
    done
done