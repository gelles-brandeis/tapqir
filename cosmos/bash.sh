#!/bin/bash

# GraceArticlePol2
#python main.py --num-iter=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-iter=30000 --model=features --n-batch=32 --learning-rate=0.0003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-iter=30000 --model=features --n-batch=32 --learning-rate=0.0001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-iter=30000 --model=features --n-batch=32 --learning-rate=0.00003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --device=cuda1 --jit
#python main.py --num-iter=50000 --model=marginal-detector --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --device=cuda1

#python main.py --num-iter=30000 --model=marginal-detector --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --device=cuda1
#python main.py --num-iter=20000 --model=detector --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --device=cuda1
#python main.py --num-iter=25000 --model=supervised --n-batch=5 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-iter=150000 --model=marginal-semisupervised --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-iter=50000 --model=semisupervised --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-iter=30000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2Supervised --negative-control=GraceArticlePol2 --device=cuda1
#python main.py --num-iter=50000 --model=tracker --n-batch=32 --learning-rate=0.002 --dataset=GraceArticlePol2 --device=cuda1
##python main.py --num-iter=10000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=FL_1_1117_0OD --device=cuda1
#python main.py --num-iter=50000 --model=tracker --n-batch=32 --learning-rate=0.002 --dataset=GraceArticlePol2 --device=cuda1 --sample
python main.py --num-iter=10000 --model=tracker --n-batch=64 --learning-rate=0.005 --dataset=GraceArticlePol2Sampled --device=cuda1 --jit

# DanPol2
#python main.py --num-iter=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=DanPol2 --device=cuda0 --jit
#python cosmos.py --num-epochs=80000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=DanPol2 --cuda0

# LarryCy3sigma54
#python main.py --num-iter=20000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-iter=30000 --model=features --n-batch=32 --learning-rate=0.0003 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-iter=80000 --model=features --n-batch=32 --learning-rate=0.0001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-iter=20000 --model=features --n-batch=32 --learning-rate=0.00003 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0 --jit
#python main.py --num-iter=50000 --model=detector --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0

# LarryCy3sigma54Short
#python main.py --num-iter=50000 --model=marginal-tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=40000 --model=marginal-detector --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=5000 --model=blocked-detector --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=20000 --model=features --n-batch=16 --learning-rate=0.0003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=20000 --model=features --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=20000 --model=detector --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0

# FL_1_1117_0OD
#python cosmos.py --num-epochs=70000 --model=features --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --cuda1 --jit
#python cosmos.py --num-epochs=30000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --cuda1

# LarryCy3sigma54Supervised
#python main.py --num-iter=10000 --model=features --n-batch=3 --learning-rate=0.005 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-iter=10000 --model=features --n-batch=3 --learning-rate=0.001 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-iter=50000 --model=detector --n-batch=3 --learning-rate=0.002 --dataset=LarryCy3sigma54Supervised --device=cuda0

#python main.py --num-iter=30000 --model=marginal-detector --n-batch=3 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --device=cuda0
#python main.py --num-iter=10000 --model=blocked-detector --n-batch=3 --learning-rate=0.001 --dataset=LarryCy3sigma54Supervised --device=cuda0

#python main.py --num-iter=20000 --model=supervised --n-batch=3 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --negative-control=LarryCy3sigma54Short --device=cuda0
#python main.py --num-iter=50000 --model=marginal-semisupervised --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Supervised --negative-control=LarryCy3sigma54Short --device=cuda0

# Gracecy3Supervised
#python main.py --num-iter=40000 --model=marginal-detector --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --device=cuda0
#python main.py --num-iter=20000 --model=detector --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --device=cuda0
#python main.py --num-iter=20000 --model=supervised --n-batch=3 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-iter=150000 --model=marginal-semisupervised --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-iter=30000 --model=semisupervised --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-iter=30000 --model=tracker --n-batch=32 --learning-rate=0.003 --dataset=Gracecy3Supervised --negative-control=Gracecy3 --device=cuda0
#python main.py --num-iter=50000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=DanPol2 --device=cuda0
#python main.py --num-iter=50000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --device=cuda0 --sample
##python main.py --num-iter=30000 --model=tracker --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54 --device=cuda0 --jit
#python main.py --num-iter=100000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --negative-control=LarryCy3sigma54NegativeControl --device=cuda0
#python main.py --num-iter=30000 --model=tracker --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0
#python main.py --num-iter=100000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=Gracecy3 --device=cuda0
