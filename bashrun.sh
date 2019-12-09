#!/bin/bash
#kernprof -o ../Datasets/Irene-Cy5-TMP-DHFR/classifier-16-001.lprof -l classifier.py --num-epochs=500 --n-batch=16 --learning-rate=0.01 --dataset=Irenecy5 --cuda
#python -i run.py --num-epochs=25000 --model=v6 --n-batch=64 --learning-rate=0.0005 --dataset=DanPol2 --cuda1
#python -i run.py --num-epochs=30000 --model=junk --n-batch=16 --learning-rate=0.0005 --dataset=LarryCy3sigma54 --cuda1
#python -i run.py --num-epochs=50000 --model=v11 --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --cuda1
#python -i run.py --num-epochs=50000 --model=v13 --n-batch=16 --learning-rate=0.001 --dataset=LarryCy3sigma54 --cuda1
#python run.py --num-epochs=20000 --model=v8 --n-batch=16 --learning-rate=0.0005 --dataset=LarryCy3sigma54 --cuda1
#python -i run.py --model=v5 --n-batch=16 --learning-rate=0.0003 --dataset=LarryCy3sigma54 --cuda1
#python -i run.py --num-epochs=20000 --model=v11 --n-batch=32 --learning-rate=0.0005 --dataset=Irenecy5 --cuda0
#python -i run.py --num-epochs=100 --model=junk --n-batch=16 --learning-rate=0.0005 --dataset=DanPol2 --cuda1 running
#python run.py --num-epochs=20000 --model=v6 --n-batch=64 --learning-rate=0.0003 --dataset=Irenecy5 --cuda0
python -i run.py --num-epochs=50000 --model=v12 --n-batch=64 --learning-rate=0.005 --dataset=Gracecy3 --cuda0
#python -i run.py --num-epochs=50000 --model=v5 --n-batch=64 --learning-rate=0.0003 --dataset=Gracecy3 --cuda0
#python run.py --num-epochs=50000 --model=v8 --n-batch=64 --learning-rate=0.0003 --dataset=Gracecy3 --cuda0
#python run.py --num-epochs=10000 --model=junk --n-batch=32 --learning-rate=0.0003 --dataset=FL_1_1117_0OD --cuda1
#python run.py --num-epochs=30000 --model=v6 --n-batch=32 --learning-rate=0.0003 --dataset=FL_1_1117_0OD --cuda1
#python run.py --num-epochs=50000 --model=v8 --n-batch=32 --learning-rate=0.0003 --dataset=FL_1_1117_0OD --cuda1
