#!/bin/bash

# GraceArticlePol2
#python main.py --num-steps=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=30000 --model=features --n-batch=32 --learning-rate=0.0003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=30000 --model=features --n-batch=32 --learning-rate=0.0001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=30000 --model=features --n-batch=32 --learning-rate=0.00003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=50000 --model=marginal-detector --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --device=cuda1

#python main.py --num-steps=30000 --model=marginal-detector --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --device=cuda1
#python main.py --num-steps=20000 --model=detector --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --device=cuda1
#python main.py --num-steps=25000 --model=supervised --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-steps=150000 --model=marginal-semisupervised --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-steps=50000 --model=semisupervised --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-steps=50000 --model=tracker --n-batch=64 --learning-rate=0.005 --dataset=GraceArticlePol2 --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.006 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1_1117_0OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1118_2225_0p3OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_2226_3338_0p6OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_3339_4444_0p8OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=FL_4445_5554_1p1OD --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_5555_6684_1p3OD --device=cuda0 --jit

# DanPol2
#python main.py --num-steps=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=DanPol2 --device=cuda0 --jit
#python cosmos.py --num-epochs=80000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=DanPol2 --cuda0

# LarryCy3sigma54
#python main.py --num-steps=20000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=30000 --model=features --n-batch=32 --learning-rate=0.0003 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=80000 --model=features --n-batch=32 --learning-rate=0.0001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=20000 --model=features --n-batch=32 --learning-rate=0.00003 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=50000 --model=detector --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0

# LarryCy3sigma54Short
#python main.py --num-steps=50000 --model=marginal-tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=40000 --model=marginal-detector --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=5000 --model=blocked-detector --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=20000 --model=features --n-batch=16 --learning-rate=0.0003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=20000 --model=features --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=20000 --model=detector --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0

# FL_1_1117_0OD
#python cosmos.py --num-epochs=70000 --model=features --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --cuda1 --jit

# LarryCy3sigma54Supervised
#python main.py --num-steps=10000 --model=features --n-batch=3 --learning-rate=0.005 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-steps=10000 --model=features --n-batch=3 --learning-rate=0.001 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-steps=50000 --model=detector --n-batch=3 --learning-rate=0.002 --dataset=LarryCy3sigma54Supervised --device=cuda0

#python main.py --num-steps=30000 --model=marginal-detector --n-batch=3 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-steps=10000 --model=blocked-detector --n-batch=3 --learning-rate=0.001 --dataset=LarryCy3sigma54Supervised --device=cuda0

#python main.py --num-steps=20000 --model=supervised --n-batch=3 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --negative-control=LarryCy3sigma54Short --device=cuda0
#python main.py --num-steps=50000 --model=marginal-semisupervised --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --negative-control=LarryCy3sigma54Short --device=cuda0

# Gracecy3Supervised
#python main.py --num-steps=40000 --model=marginal-detector --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --device=cuda0
#python main.py --num-steps=20000 --model=detector --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --device=cuda0
#python main.py --num-steps=20000 --model=supervised --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-steps=150000 --model=marginal-semisupervised --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-steps=30000 --model=semisupervised --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=DanPol2 --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --device=cuda0 --sample
##python main.py --num-steps=5000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0 --jit
##python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0 --jit --sample
#python main.py --num-steps=10000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Sampled --device=cuda0 --jit
#python main.py --num-steps=100000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1_1117_0OD_atten --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1118_2225_0p3OD_atten --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_2226_3338_0p6OD_atten --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_3339_4444_0p8OD_atten --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0015 --dataset=FL_4445_5554_1p1OD_atten --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_5555_6684_1p3OD_atten --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0015 --dataset=FL_4445_5554_1p1OD --device=cuda1 --jit
#python main.py --num-steps=1000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_5555_6684_1p3OD --device=cuda1 --jit
#python main.py --num-steps=30000 --model=tracker --n-batch=16 --learning-rate=0.0051 --dataset=LarryCy3sigma54Short --device=cuda0
#python main.py --num-steps=15000 --model=tracker --n-batch=10 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=GraceArticlePol2NegativeControl --device=cuda1
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.0045 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=10000 --model=tracker --n-batch=8 --learning-rate=0.0045 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=FL_1118_2225_0p3OD --device=cuda0
#python main.py --num-steps=50000 --model=tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0 --jit
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.0045 --dataset=DanPol2 --negative-control=DanPol2NegativeControl --device=cuda1 --jit
#python main.py --num-steps=5000 --model=tracker --n-batch=32 --learning-rate=0.0059 --dataset=GraceArticleSpt5 --negative-control=GraceArticleSpt5NegativeControl --device=cuda1
#python main.py --num-steps=15000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=DanZhouPol2 --negative-control=DanZhouPol2NegativeControl --device=cuda0 --jit
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.004 --dataset=FL_1_1117_0OD --negative-control=FL_1_1117_0OD.NegativeControl --device=cuda1 --jit
#python main.py --num-steps=30000 --model=tracker --n-batch=32 --learning-rate=0.0035 --dataset=FL_1118_2225_0p3OD --negative-control=FL_1118_2225_0p3OD.NegativeControl --device=cuda0 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0037 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0044 --dataset=GraceArticleSpt5 --negative-control=GraceArticleSpt5NegativeControl --device=cuda1 --jit
#python main.py --num-steps=20000 --model=tracker --n-batch=16 --learning-rate=0.0044 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda0 --jit
