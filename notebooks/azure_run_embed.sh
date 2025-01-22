#!/bin/sh


echo hello >> test.txt
which python >> test.txt

python -u embed_EMS.py --seting True --model 1 --cc  1  --file "anti" --token 1 >> test1.txt & 
python -u embed_EMS.py --seting True --model 1 --cc  1  --file "anti" --token 2 >> test2.txt &
python -u embed_EMS.py --seting True --model 1 --cc  1  --file "anti" --token 0 >> test3.txt &

# "Chain choice. 0: sep, 1: together %(default)", 
# file "covi" or "anti2

#"token for linking 0:2,[cls,G*20,GGGGS*3] %(default)", 

echo done