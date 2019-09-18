#!/bin/bash
#kernprof -o ../Datasets/Irene-Cy5-TMP-DHFR/classifier-16-001.lprof -l classifier.py --num-epochs=500 --n-batch=16 --learning-rate=0.01 --dataset=Irenecy5 --cuda
#python -i run.py --num-epochs=100 --model=feature --n-batch=16 --learning-rate=0.001 --dataset=FL_3339_4444_0p8OD --cuda0
python -i run.py --num-epochs=100 --model=feature --n-batch=16 --learning-rate=0.001 --dataset=Irenecy5 --cuda0
#python -i run.py --num-epochs=100 --model=v4 --n-batch=16 --learning-rate=0.0005 --dataset=Orange --cuda0
#python -i classifier.py --num-epochs=100 --model=v3 --n-batch=16 --learning-rate=0.02 --dataset=Irenecy5 --cuda
