#!/bin/sh
### General options
### –- specify queue --# gpu: gpuv100, course :c27666
#BSUB -q c27666  
### -- set the job Name --
#BSUB -J embed_EMS
### -- ask for number of cores (default: 1) --
#BSUB -n 8
# ### -- Select the resources: 1 gpu in exclusive process mode --
# #BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:30
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s184243@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J.out
#BSUB -e Error%J.err

# here follow the commands you want to execute

#module load python3/3.8.2
#source /zhome/e7/a/137819/deep_env/bin/activate
#/zhome/e7/a/137819/deep_env/bin/python3 -u ./code/WIP_model.py >> report.txt

source /zhome/e7/a/137819/miniconda3/bin/activate

> report.txt
echo hello >> report.txt
which python >> report.txt

#which conda >> report.txt
#conda activate tranform_env
#echo hello2 >> report.txt

conda activate gpu_tranform >> report.txt

# which python >> report.txt

#command time python -u  clean_model.py --epocs = $1  --ll = $2 --heads = $3 --LR = $4 --warmup = $5  >> report.txt
# command time python -u  clean_model.py --epocs = 50  --ll = 8 --heads = 4 --LR = 1e-05 --warmup = 4000  >> report.txt


python -u embed_EMS.py --run_local False --file cova --model 4 --cc 1 >> report.txt



echo done >> report.txt





##source /zhome/06/a/147115/BSc_venv/bin/activate


##/zhome/06/a/147115/BSc_venv/bin/python3 -u /zhome/06/a/147115/02456_project_group_72/src/chatbot.py > /zhome/06/a/147115/02456_project_group_72/src/logs/cb_model6/output.txt

