#!/bin/bash

# DanPol2

#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.0045 --dataset=DanPol2 --negative-control=DanPol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.0059 --dataset=GraceArticleSpt5 --negative-control=GraceArticleSpt5NegativeControl --device=cuda1
#python main.py --num-steps=15000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=DanZhouPol2 --negative-control=DanZhouPol2NegativeControl --device=cuda0
#cosmos --num-steps=20000 --model=tracker --n-batch=8 --learning-rate=0.005 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda1
#cosmos --num-steps=25000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda1
#cosmos --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --device=cuda1
cosmos --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2NegativeControl1 --negative-control=GraceArticlePol2NegativeControl2 --device=cuda1
#cosmos --num-steps=30000 --model=tracker --n-batch=10 --learning-rate=0.005 --dataset=LarryCy3sigma54NegativeControl1 --negative-control=LarryCy3sigma54NegativeControl2 --device=cuda0
