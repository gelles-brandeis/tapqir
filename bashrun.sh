#!/bin/bash
#kernprof -o ../Datasets/Irene-Cy5-TMP-DHFR/classifier-16-001.lprof -l classifier.py --num-epochs=500 --n-batch=16 --learning-rate=0.01 --dataset=Irenecy5 --cuda

# LarryCy3sigma54
#python -i run.py --num-epochs=10000 --model=v12 --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.01 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.0003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=features --n-batch=16 --learning-rate=0.00003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=detector --n-batch=16 --learning-rate=0.003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=80000 --model=tracker --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=detector --n-batch=16 --learning-rate=0.0003 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1
#python run.py --num-epochs=25000 --model=detector --n-batch=16 --learning-rate=0.0001 --dataset=LarryCy3sigma54Short --negative-control=LarryCy3sigma54NegativeControlShort --cuda1

#python -i run.py --model=features --n-batch=16 --learning-rate=0.0005 --dataset=LarryCy3sigma54 --cuda1
#python -i run.py --num-epochs=20000 --model=v11 --n-batch=32 --learning-rate=0.0005 --dataset=Irenecy5 --cuda0
#python -i run.py --num-epochs=30000 --model=v13 --n-batch=64 --learning-rate=0.0003 --dataset=Gracecy3 --cuda0
#python -i run.py --num-epochs=30000 --model=v11 --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --cuda1

# GraceArticlePol2
#python run.py --num-epochs=20000 --model=features --n-batch=32 --learning-rate=0.01 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=20000 --model=features --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=30000 --model=detector --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=40000 --model=tracker --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=20000 --model=features --n-batch=32 --learning-rate=0.0003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=20000 --model=features --n-batch=32 --learning-rate=0.0001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=10000 --model=detector --n-batch=32 --learning-rate=0.003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=10000 --model=detector --n-batch=32 --learning-rate=0.001 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0
#python run.py --num-epochs=10000 --model=detector --n-batch=32 --learning-rate=0.0003 --dataset=GraceArticlePol2 --negative-control=GraceArticlePol2NegativeControl --cuda0

# FL_1_1117_0OD
#python -i run.py --num-epochs=10000 --model=v13 --n-batch=32 --learning-rate=0.0005 --dataset=FL_1_1117_0OD --cuda1
python run.py --num-epochs=50000 --model=features --n-batch=32 --learning-rate=0.001 --dataset=FL_1_1117_0OD --cuda1
