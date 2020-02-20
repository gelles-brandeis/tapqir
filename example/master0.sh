#!/bin/bash

#python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --negative-control=FL_1_1117_0OD.NegativeControl --device=cuda0
python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_1118_2225_0p3OD --negative-control=FL_1118_2225_0p3OD.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.01 --dataset=FL_2226_3338_0p6OD --negative-control=FL_2226_3338_0p6OD.NegativeControl --device=cuda0
#python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.01 --dataset=FL_3339_4444_0p8OD --negative-control=FL_3339_4444_0p8OD.NegativeControl --device=cuda0
#python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.01 --dataset=FL_4445_5554_1p1OD --negative-control=FL_4445_5554_1p1OD.NegativeControl --device=cuda0
#python main.py --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.01 --dataset=FL_5555_6684_1p3OD --negative-control=FL_5555_6684_1p3OD.NegativeControl --device=cuda0
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda0
