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


# In[2]:


model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")


# ## 

# ## Data Load

# In[3]:


work_dir = os.getcwd()
data_dir = os.path.join(work_dir, '../data')


# In[4]:


data = pd.read_excel(os.path.join(data_dir, 'external/antibody_info.xlsx'), header=1)
display(data)


# In[5]:


df = data[["Antibody  Name","Heavy chain AA","Light chain AA"]]

df = pd.melt(df, id_vars="Antibody  Name", value_vars=["Heavy chain AA","Light chain AA"],var_name = "Chain ID",value_name = "chain AA")

df["identifyer"] = df["Antibody  Name"].astype(str) + " / " + df["Chain ID"]




df = df[["identifyer","chain AA"]]
# 

df = list(zip(*map(df.get, df[["identifyer","chain AA"]])))

# display(df)


# ## Embed 

# #### ESM 650M

# In[ ]:


model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results


# In[14]:


# # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]
data = df#[0:10]

batch_labels, batch_strs, batch_tokens = batch_converter(data)

batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
# only the tokens are given the model, 
# repr_layers only returns the 33 layers as representtion for every amino aids
# return_contacts predicts contracts between AA
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

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


# In[60]:





# In[ ]:


import pickle 

with open("../data/interim/embed_EMS_650_seperate", 'wb') as f:
    pickle.dump(results["representations"][33], f)

