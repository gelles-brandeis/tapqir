#!/bin/bash

python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.006 --dataset=FL_1_1117_0OD_atten --negative-control=FL_1_1117_0OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=FL_1118_2225_0p3OD_atten --negative-control=FL_1118_2225_0p3OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=FL_2226_3338_0p6OD_atten --negative-control=FL_2226_3338_0p6OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=FL_3339_4444_0p8OD_atten --negative-control=FL_3339_4444_0p8OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=FL_4445_5554_1p1OD_atten --negative-control=FL_4445_5554_1p1OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=25000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=FL_5555_6684_1p3OD_atten --negative-control=FL_5555_6684_1p3OD_atten.NegativeControl --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=10 --learning-rate=0.005 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda1
