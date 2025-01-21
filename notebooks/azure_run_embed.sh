#!/bin/sh
>test1.txt

echo hello >> test.txt
which python >> test.txt

python -u embed_EMS.py --model 3 --cc 1 >> test5.txt 


# python -u embed_EMS.py --model 4 --cc 1 >> test1.txt &


echo done