#!/bin/bash

python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --negative-control=FL_1_1117_0OD.NegativeControl --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_1118_2225_0p3OD --device=cuda1
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_1118_2225_0p3OD --negative-control=FL_1118_2225_0p3OD.NegativeControl --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_2226_3338_0p6OD --negative-control=FL_2226_3338_0p6OD.NegativeControl --device=cuda0
