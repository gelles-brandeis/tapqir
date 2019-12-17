#!/bin/bash

# GraceArticlePol2
#python cosmos.py --num-epochs=30000 --model=detector --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
python cosmos.py --num-epochs=50000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0

# LarryCy3sigma54Short
#python cosmos.py --num-epochs=30000 --model=detector --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0

# FL_1_1117_0OD
#python cosmos.py --num-epochs=70000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=FL_1_1117_0OD --cuda1
#python cosmos.py --num-epochs=70000 --model=detector --n-batch=32 --learning-rate=0.001 --dataset=FL_1_1117_0OD --cuda1
