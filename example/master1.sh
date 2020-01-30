#!/bin/bash
python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_3339_4444_0p8OD --negative-control=FL_3339_4444_0p8OD.NegativeControl --device=cuda1
python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_4445_5554_1p1OD --negative-control=FL_4445_5554_1p1OD.NegativeControl --device=cuda1
python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_5555_6684_1p3OD --negative-control=FL_5555_6684_1p3OD.NegativeControl --device=cuda1
