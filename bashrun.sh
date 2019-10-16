#!/bin/bash
#kernprof -o ../Datasets/Irene-Cy5-TMP-DHFR/classifier-16-001.lprof -l classifier.py --num-epochs=500 --n-batch=16 --learning-rate=0.01 --dataset=Irenecy5 --cuda
python -i run.py --num-epochs=100 --model=v5 --n-batch=16 --learning-rate=0.0005 --dataset=LarryCy3sigma54 --cuda0
#python -i run.py --num-epochs=100 --model=junk --n-batch=16 --learning-rate=0.001 --dataset=Gracecy3 --cuda0
#python -i run.py --num-epochs=100 --model=v4 --n-batch=16 --learning-rate=0.002 --dataset=Gracecy3 --cuda1
#python -i run.py --num-epochs=100 --model=v4 --n-batch=16 --learning-rate=0.001 --dataset=DanPol2 --cuda0
#python -i classifier.py --num-epochs=100 --model=v3 --n-batch=16 --learning-rate=0.02 --dataset=Irenecy5 --cuda
