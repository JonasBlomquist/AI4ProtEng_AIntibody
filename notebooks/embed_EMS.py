#!/usr/bin/env python
# coding: utf-8

# # Classification

# ## Imports

# In[1]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import esm
import math


# In[2]:

# ## augments


import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Augments for embeding.")

# Add arguments
parser.add_argument('--model', 
                    type=int, 
                    help="index of model to use from ESM 0:5: %(default)", 
                    default=1)
parser.add_argument('--cc', 
                    type=int, 
                    help="Chain choice. 0: sep, 1: together %(default)", 
                    default=0)

parser.add_argument('--file', 
                    type=str, 
                    help="file to embed %(default)", 
                    default="something")
parser.add_argument('--run_local', 
                    type=bool, 
                    help="to run local  %(default)", 
                    default=True)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
# name = args.name
model_args = args.model
cc_args = args.cc
file_args = args.file

local_run = args.run_local




#%%

if local_run :
    print("using local seting")
    model_args = 4
    cc_args = 1 # 0: sep, 1 together



if model_args not in range(6):
    print("model index not valid")
    if cc_args not in range(2):
        print("cc choince not valid")
        sys.exit()






# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")



# ## 

# ## Data Load

# In[3]:


work_dir = os.getcwd()
data_dir = os.path.join(work_dir, '../data')


# In[4]:


data = pd.read_excel(os.path.join(data_dir, 'external/antibody_info.xlsx'), header=1)
# display(data)


# In[5]:


#chain mode,
# Either seporate embeding or together by cls token as linker
# Seporate heavy chains is frst.

chain_choice = ["seperate","together"]
chain_mode = chain_choice[cc_args]
print(f"chain mode: {chain_mode} \n")


# get the name and chains
df = data[["Antibody  Name","Heavy chain AA","Light chain AA"]]

# embed as septer
if cc_args == 0: 

    print("________seting data long/seperate________\n\n")
    # pivot longer
    df = pd.melt(df, id_vars="Antibody  Name", value_vars=["Heavy chain AA","Light chain AA"],var_name = "Chain ID",value_name = "chain AA")
    # new identy colun, combi of antibody and chain name
    df["identifyer"] = df["Antibody  Name"].astype(str) + " / " + df["Chain ID"]

    # subset to mine colummn
    df = df[["identifyer","chain AA"]]


else: 
    # embed together with cls as divider

    # Token to insert in the middle
    link_token = "<cls>"

    # concat wtih toek
    print(f"______seting data together with: {link_token}______\n\n")


    # Combine the columns with the token in between
    df['chain AA'] = df['Heavy chain AA'] + link_token + df['Light chain AA']

    # lazy code
    df["identifyer"] = df["Antibody  Name"]


    # subset to relevant colmun
    df = df[["identifyer","chain AA"]]


# format to esm, list of tubles with (name, seq) 
df = list(zip(*map(df.get, df[["identifyer","chain AA"]])))

# ## Embed 

# #### ESM 650M

# In[ ]:

print("down/load model\n")
torch.hub.set_dir(data_dir)

esm2_model_names = [
    'esm2_t6_8M_UR50D',    # ESM-2 8M model     # 0 
    'esm2_t12_35M_UR50D',   # ESM-2 35M model   # 1
    'esm2_t30_150M_UR50D',  # ESM-2 150M model  # 2
    'esm2_t33_650M_UR50D',  # ESM-2 650M model  # 3
    'esm2_t36_3B_UR50D',    # ESM-2 3B model    # 4 
    'esm2_t48_15B_UR50D'    # ESM-2 15B model   # 5
]

model_name = esm2_model_names[model_args]
print(f"model to use: {model_name}")

model_path = data_dir+"/checkpoints/"+model_name+".pt"

if True == os.path.isfile(model_path):
    # local file
    print("model found  in local file")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
else:
    # downloading
    print("download file")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    

# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
# model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
# model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()




batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results




# In[14]:

batch_size = 10

# data format needs to follow
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]

data = df#[0:10]  # for testing local



# get label name, string length and boken of sequence
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# used for sequence representaion.
# batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

last_layer= sum(1 for layer in model.modules() if "trans" in str(type(layer)).lower())
print("embeding")

# if torch.cuda.is_available():
#     batch_tokens = batch_tokens.cuda()
#     model = model.cuda()

all_embeddings = []
all_contacts = []


number_batches = math.ceil(len(df)/batch_size)

#local_test
# number_batches = 2

for batch_index in range(number_batches):
    if batch_index % 10 ==0:
        print(f"{(batch_index/number_batches)*100} % done")

    batch_to_run = batch_tokens[batch_index*batch_size:(batch_index+1)*batch_size]

    # Extract per-residue representations (on CPU)
    # only the tokens are given the model, 
    # repr_layers only returns the 33 layers as representtion for every amino aids
    # return_contacts predicts contracts between AA

    with torch.no_grad():
        results = model(batch_to_run, repr_layers=[last_layer], return_contacts=True)
    del results["logits"]
    del results["attentions"]


    # Extract the embeddings from the model output (layer 33 for ESM-1b)
    embeddings = results['representations'][last_layer]  # Choose layer 33 for the embeddings

    # Concatenate embeddings for this batch
    all_embeddings.append(embeddings)
    all_contacts.append(results['contacts'])

# Concatenate embeddings from all batches into a single tensor
concatenated_embeddings = torch.cat(all_embeddings, dim=0)
concatenated_contacts = torch.cat(all_contacts, dim=0)



results = {"contacts" : concatenated_contacts, 
           "representations" : {last_layer :concatenated_embeddings}}







# # Generate per-sequence representations via averaging
# # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
# sequence_representations = []
# for i, tokens_len in enumerate(batch_lens):
#     sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))


# attentions scored for pos to pos
# # Look at the unsupervised self-attention map contact predictions
# import matplotlib.pyplot as plt
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(seq)
#     plt.show()




# In[ ]:

model_name.split("_")

import pickle 
print("saving")
with open(f"../data/interim/embed_EMS_{model_name.split("_")[2]}_{chain_mode}", 'wb') as f:
    pickle.dump(results, f)


print("done")

# %%
