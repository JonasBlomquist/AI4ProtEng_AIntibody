#!/bin/sh

# echo hello >> test.txt
# which python >> test.txt

# t= which conda
# echo $t
# conda activate aipro


k= which python
echo  $k
for model in 2
do

# >test1.txt
# python -u embed_EMS.py --seting True --model $model --cc  0  --file "anti"  >> test1.txt & 


for linker in 1 2 
do

echo "hello"
>test1.txt
python -u embed_EMS.py --seting True --model $model --cc  1  --file "anti" --token $linker >> test1.txt  &
    
done
done

# >test1.txt
# python -u embed_EMS.py --seting True --model 1 --cc  1  --file "anti" --token 1 >> test1.txt & 
# >test2.txt
# python -u embed_EMS.py --seting True --model 1 --cc  1  --file "anti" --token 2 >> test2.txt &
# >test3.txt
# python -u embed_EMS.py --seting True --model 1 --cc  0   >> test5.txt &

# "Chain choice. 0: sep, 1: together %(default)", 
# file "covi" or "anti2

#"token for linking 0:2,[cls,G*20,GGGGS*3] %(default)", 

echo done