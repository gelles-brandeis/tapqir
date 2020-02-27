#!/bin/bash

#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1_1117_0OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1118_2225_0p3OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_2226_3338_0p6OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_3339_4444_0p8OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=FL_4445_5554_1p1OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_5555_6684_1p3OD --device=cuda0 --jit

# DanPol2

#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=DanPol2 --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --device=cuda0 --sample
##python main.py --num-steps=5000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0 --jit
##python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0 --jit --sample
#python main.py --num-steps=10000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Sampled --device=cuda0 --jit
#python main.py --num-steps=100000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0

#python main.py --num-steps=10000 --model=tracker --n-batch=8 --learning-rate=0.0045 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1118_2225_0p3OD --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0 --jit
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.0045 --dataset=DanPol2 --negative-control=DanPol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.0059 --dataset=GraceArticleSpt5 --negative-control=GraceArticleSpt5NegativeControl --device=cuda1
#python main.py --num-steps=15000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=DanZhouPol2 --negative-control=DanZhouPol2NegativeControl --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.0055 --dataset=FL_1_1117_0OD --negative-control=FL_1_1117_0OD.NegativeControl --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.0055 --dataset=FL_1118_2225_0p3OD --negative-control=FL_1118_2225_0p3OD.NegativeControl --device=cuda0
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_2226_3338_0p6OD --negative-control=FL_2226_3338_0p6OD.NegativeControl --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=10 --learning-rate=0.005 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda1
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0043 --dataset=GraceArticleSpt5 --negative-control=GraceArticleSpt5NegativeControl --device=cuda0
#python main.py --num-steps=30000 --model=marginal --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --device=cuda1
cosmos --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=FL_3339_4444_0p8OD --negative-control=FL_3339_4444_0p8OD.NegativeControl --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda1
#cosmos --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --device=cuda0
#cosmos --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2NegativeControl2 --negative-control=GraceArticlePol2NegativeControl1 --device=cuda1
