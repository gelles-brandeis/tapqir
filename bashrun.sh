#!/bin/bash
#kernprof -o ../Datasets/Irene-Cy5-TMP-DHFR/classifier-16-001.lprof -l classifier.py --num-epochs=500 --n-batch=16 --learning-rate=0.01 --dataset=Irenecy5 --cuda
python -i run.py --num-epochs=20000 --model=v6 --n-batch=64 --learning-rate=0.0005 --dataset=DanPol2 --cuda1
#python run.py --num-epochs=20000 --model=v6 --n-batch=24 --learning-rate=0.0003 --dataset=LarryCy3sigma54 --cuda0
#python -i run.py --num-epochs=20000 --model=v6 --n-batch=32 --learning-rate=0.0005 --dataset=Irenecy5 --cuda0
#python -i run.py --num-epochs=100 --model=junk --n-batch=16 --learning-rate=0.0005 --dataset=DanPol2 --cuda1 running
#python run.py --num-epochs=20000 --model=v6 --n-batch=64 --learning-rate=0.0003 --dataset=Irenecy5 --cuda0
#python run.py --num-epochs=20000 --model=v6 --n-batch=64 --learning-rate=0.0003 --dataset=Gracecy3 --cuda1
