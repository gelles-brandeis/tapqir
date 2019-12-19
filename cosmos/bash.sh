#!/bin/bash

# GraceArticlePol2
#python cosmos.py --num-epochs=50000 --model=features --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0 --jit
#python cosmos.py --num-epochs=100000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python cosmos.py --num-epochs=100000 --model=tracker --n-batch=32 --learning-rate=0.005 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0

# DanPol2
#python cosmos.py --num-epochs=80000 --model=features --n-batch=32 --learning-rate=0.005 --dataset=DanPol2 --cuda0 --jit
#python cosmos.py --num-epochs=80000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=DanPol2 --cuda0

# LarryCy3sigma54Short
python main.py --num-iter=20000 --model=features --n-batch=16 --learning-rate=0.005 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --device=cuda0

# FL_1_1117_0OD
#python cosmos.py --num-epochs=70000 --model=features --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --cuda1 --jit
#python cosmos.py --num-epochs=30000 --model=detector --n-batch=32 --learning-rate=0.005 --dataset=FL_1_1117_0OD --cuda1
