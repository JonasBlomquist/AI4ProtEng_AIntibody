#!/usr/bin/env python
# coding: utf-8

# In[16]:


# # Classification

# ## Imports


# In[17]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import esm
import math


# In[26]:


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
                    default="anti")


parser.add_argument('--token', 
                    type=int, 
                    help="token for linking 0:2,[cls,G*20,GGGGS*3] %(default)", 
                    default=0)


parser.add_argument('--seting', 
                    type=bool, 
                    help="Use seting or not  %(default)", 
                    default=False)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
# name = args.name
model_args = args.model
cc_args = args.cc
file_args = args.file
token_args = args.token

run_seting = args.seting




# In[33]:


if False == run_seting :
    print("using local seting")

    ##### model options
    # ESM-2 8M model     # 0 
    # ESM-2 35M model   # 1
    # ESM-2 150M model  # 2
    # ESM-2 650M model  # 3
    # ESM-2 3B model    # 4 
    # ESM-2 15B model   # 5
    model_args = 0


    ##### chain choce 
    # 0: sep, 
    # 1 together

    # cc_args = 0  
    cc_args = 1     

    # file choice
    # anti or cova as string
    file_args = "anti"
    # file_args = "cova" 


    token_args = 0

    chains = ["sep","tog"][cc_args]
    print(f"___{file_args=}__\n__{model_args=}___\n___chinas as: {chains=}___\n")



if model_args not in range(6):
    print("model index not valid")
    if cc_args not in range(2):
        print(f"cc choince not valid")
        if file_args not in ["anti","cova"]:
            print("lol")
            # sys.exit()






# model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")



# ## 

# ## Data Load


# In[187]:


work_dir = os.getcwd()
data_dir = os.path.join(work_dir, '../data')

if file_args == "anti":
    print("label file in excel format")
    data = pd.read_excel(os.path.join(data_dir, 'external/antibody_info.xlsx'), header=1)
elif file_args == "cova":
    print("unlabel file in csv format")
    data = pd.read_csv(os.path.join(data_dir, 'external/covabdab_search_results.csv'))
else:
    print("work in prgoess:")






# In[191]:


### filter
if file_args == "cova":
    print("filtering for VL chains 'nan' ")
    print("nan in these index")
    where_na = data[data['VL'].isna()].index
    print(where_na)

    print(f"there were drops this many rows: {len(where_na)}")
    data = data.dropna(subset=['VL'])





# In[192]:


#### # extacting file chains to embed

if file_args == "anti":
    print("geting chains in label")
    
    # get the name and chains
    chains = data[["Antibody  Name","Heavy chain AA","Light chain AA"]]



elif file_args == "cova":
    print("geting chains in unlabel")


    # get the name and chains
    chains = data[["Name","VHorVHH","VL"]]
    chains.columns = ["Antibody  Name","Heavy chain AA","Light chain AA"]

else:
    print("work in prgoess:")


# display(df.head())


# In[193]:


#chain mode,
# Either seporate embeding or together by cls token as linker
# Seporate heavy chains is frst.

chain_choice = ["seperate","together"]
chain_mode = chain_choice[cc_args]
print(f"chain mode: {chain_mode} \n")

df = chains

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
    link_tokens = ["<cls>","G"*20,"GGGGS"*3]
    link_token = link_tokens[token_args]

    # concat wtih toek
    print(f"______seting data together with: {link_token}______\n\n")


    # Combine the columns with the token in between
    df['chain AA'] = df['Heavy chain AA'] + link_token + df['Light chain AA']

    # lazy code
    df["identifyer"] = df["Antibody  Name"]


    # subset to relevant colmun
    df = df[["identifyer","chain AA"]]


# format to esm, list of tubles with (name, seq) 
esm_input = list(zip(*map(df.get, df[["identifyer","chain AA"]])))



###
# display(esm_input[:2])



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

# Define a mapping of model arguments to their respective functions
model_mapping = {
    0: esm.pretrained.esm2_t6_8M_UR50D,
    1: esm.pretrained.esm2_t12_35M_UR50D,
    2: esm.pretrained.esm2_t30_150M_UR50D,
    3: esm.pretrained.esm2_t33_650M_UR50D,
    4: esm.pretrained.esm2_t36_3B_UR50D,
    5: esm.pretrained.esm2_t48_15B_UR50D,
}

# Function to get the model
def get_model(model_args):
    if model_args in model_mapping:
        return model_mapping[model_args]()
    else:
        raise ValueError(f"Invalid model_args: {model_args}. Must be one of {list(model_mapping.keys())}")

model_name = esm2_model_names[model_args]

model_path = data_dir+"/checkpoints/"+model_name+".pt"

if True == os.path.isfile(model_path):
    # local file
    print("model found  in local file")
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_path)
else:
    # downloading
    print("download file")
    model, alphabet = get_model(model_args)


# Push model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("Model moved to GPU \n")
else:
    print("no cuda 4 u poor mann \n")

model_name = esm2_model_names[model_args]
print(f"model to use: {model_name}")



batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results




# In[ ]:


batch_size = 10

# data format needs to follow
# data = [
#     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
#     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
#     ("protein3",  "K A <mask> I S Q"),
# ]


# 
data_esm = esm_input#[0:10]  # for testing local



# get label name, string length and boken of sequence
batch_labels, batch_strs, batch_tokens = batch_converter(data_esm)




# used for sequence representaion.
# batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

last_layer= sum(1 for layer in model.modules() if "trans" in str(type(layer)).lower())
print("embeding")

# if torch.cuda.is_available():
#     batch_tokens = batch_tokens.cuda()
#     model = model.cuda()

all_embeddings = []
all_contacts = []
all_atten = []

number_batches = math.ceil(len(df)/batch_size)

#local_test
# number_batches = 2

for batch_index in range(number_batches):


    if batch_index % 10 ==0:
        print(f"{(batch_index/number_batches)*100} % done")

    # batch_to_run = batch_tokens[batch_index*batch_size:(batch_index+1)*batch_size]
    batch_to_run = batch_tokens[batch_index * batch_size:(batch_index + 1) * batch_size]

        
    # Push tokens to GPU
    if torch.cuda.is_available():
        batch_to_run = batch_to_run.cuda()


    # Extract per-residue representations (on CPU)
    # only the tokens are given the model, 
    # repr_layers only returns the 33 layers as representtion for every amino aids
    # return_contacts predicts contracts between AA

    with torch.no_grad():
        results = model(batch_to_run, repr_layers=[last_layer], return_contacts=True)
    del results["logits"]
    del results["attentions"]

    if torch.cuda.is_available():
        # results = results.to("cpu")
        print("Embedding data...")
        print(results['representations'][last_layer].device)
        print(results['contacts'].device)
        
    torch.cuda.empty_cache()

    # Extract the embeddings from the model output (layer 33 for ESM-1b)
    embeddings = results['representations'][last_layer].to("cpu")  # Choose layer 33 for the embeddings
    contrast = results['contacts'].to("cpu")

    # Concatenate embeddings for this batch
    all_embeddings.append(embeddings)
    all_contacts.append(contrast)
    # all_atten.append(results["attentions"])

    del results


    if torch.cuda.is_available():
        # Get GPU memory usage
        total_memory = torch.cuda.get_device_properties(0).total_memory  # Total GPU memory
        reserved_memory = torch.cuda.memory_reserved(0)  # Reserved by PyTorch
        allocated_memory = torch.cuda.memory_allocated(0)  # Allocated by tensors
        free_memory = reserved_memory - allocated_memory  # Free inside reserved memory

        print(f"Total memory: {total_memory / 1024**2:.2f} MB")
        print(f"Reserved memory: {reserved_memory / 1024**2:.2f} MB")
        print(f"Allocated memory: {allocated_memory / 1024**2:.2f} MB")
        print(f"Free memory: {free_memory / 1024**2:.2f} MB")

# Concatenate embeddings from all batches into a single tensor
concatenated_embeddings = torch.cat(all_embeddings, dim=0)
concatenated_contacts = torch.cat(all_contacts, dim=0)
# concatenated_atten = torch.cat(all_atten, dim=0)


print("model done")
results = {"contacts" : concatenated_contacts, 
           "representations" : {last_layer :concatenated_embeddings,},}
        #    "attentions" : concatenated_atten}







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


model_used = model_name.split("_")[2]

import pickle 
print("saving")

save_file = ""
if file_args =="cova":
    save_file = "cova_"

if cc_args == 1:
    # Token to insert in the middle
    link_tokens = ["_<cls>","_G","_GGGGS"]
    save_token = link_tokens[token_args]
else:
    save_token = ""

with open(f"../data/interim/{save_file}embed_EMS_{model_used}_{chain_mode}{save_token}", 'wb') as f:
    pickle.dump(results, f)


print("done")


# %%





import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = [
    [0.36656891495601174, '8M', 'together', 'full_seq', '<cls>'],
    [0.36803519061583588, '8M', 'together', 'cdr', '<cls>'],
    [0.36070381231671556, '8M', 'together', 'full_seq', '_G'],
    [0.37536656891495680, '8M', 'together', 'cdr', '_G'],
    [0.35043988269794724, '8M', 'together', 'full_seq', '_GGGGS'],
    [0.36950146627565983, '8M', 'together', 'cdr', '_GGGGS'],
    [0.32991202346041054, '8M', 'separate', 'full_seq', '_GGGGS'],
    [0.35777126099706746, '8M', 'separate', 'cdr', '_GGGGS'],
    [0.36656891495601174, '35M', 'together', 'full_seq', '<cls>'],
    [0.37829912023460413, '35M', 'together', 'cdr', '<cls>'],
    [0.35043988269794724, '35M', 'together', 'full_seq', '_G'],
    [0.36803519061583583, '35M', 'together', 'cdr', '_G'],
    [0.36950146627565983, '35M', 'together', 'full_seq', '_GGGGS'],
    [0.38269794721407624, '35M', 'together', 'cdr', '_GGGGS'],
    [0.34457478005865105, '35M', 'separate', 'full_seq', '_GGGGS'],
    [0.38709677419354843, '35M', 'separate', 'cdr', '_GGGGS'],
    [0.35630498533724341, '150M', 'together', 'full_seq', '<cls>'],
    [0.37976539589442815, '150M', 'together', 'cdr', '<cls>'],
    [0.36363636363636365, '150M', 'together', 'full_seq', '_G'],
    [0.37096774193548391, '150M', 'together', 'cdr', '_G'],
    [0.34310850439882695, '150M', 'together', 'full_seq', '_GGGGS'],
    [0.37683284457478006, '150M', 'together', 'cdr', '_GGGGS'],
    [0.35190615835777131, '150M', 'separate', 'full_seq', '_GGGGS'],
    [0.36510263929618771, '150M', 'separate', 'cdr', '_GGGGS'],
    [0.3944281524926686, '3B', 'together', 'full_seq', '<cls>'],
    [0.41055718475073316, '3B', 'together', 'cdr', '<cls>'],
    [0.37976539589442815, '3B', 'separate', 'full_seq', '<cls>'],
    [0.40762463343108507, '3B', 'separate', 'cdr', '<cls>'],
    [0.28152492668621704, 'blum', '', 'full_seq', ''],
    [0.23900293255131966, 'blum', '', 'cdr', '']
]

newdata=[]

for index, line in enumerate(data):
    lineappend=line
    if line[2] == "separate":
        lineappend = line[:4]
        # print(line)
        lineappend.append("")
    newdata.append(lineappend)

newdata = data
# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['Value', 'M', 'Type', 'SeqType', 'Label'])

# Set up plot
plt.figure(figsize=(10, 6))

# Define colors for different linkers
linker_colors = {
    '_G': 'green',
    '_GGGGS': 'orange',
    '<cls>': 'blue',
}

# Define a color for 'cdr' (for the line)
cdr_line_color = 'red'
other_line_color = 'black'

# Plot each combination of SeqType, Type, and M
for (seq_type, typ) in df[['SeqType', 'Type']].drop_duplicates().values:
    subset = df[(df['SeqType'] == seq_type) & (df['Type'] == typ)]
    
    # If it's a 'together' type, split by 'Label' and color the points
    if typ == 'together':
        for label in subset['Label'].unique():
            label_subset = subset[subset['Label'] == label]
            color = linker_colors.get(label, 'blue')  # Use color based on the linker
            linestyle = 'dashed' if 'cdr' in label else 'solid'  # Dash lines for cdr
            plt.plot(label_subset['M'], label_subset['Value'], 
                     label=f'{seq_type} - {typ} - {label}', 
                     marker='o', linestyle=linestyle, color=color)
    else:
        # Plot 'separate' as usual, with color for cdr line
        color = linker_colors.get(subset['Label'].values[0], 'blue')  # Use color based on linker
        linestyle = 'solid' if 'cdr' not in subset['Type'].values[0] else 'dashed'  # Line style for cdr
        plt.plot(subset['M'], subset['Value'], label=f'{seq_type} - {typ}', marker='o', color=color, linestyle=linestyle)

# Customize plot
plt.title('Plot of Values by M Value')
plt.xlabel('M Values')
plt.ylabel('Value')

# Adjust legend to handle the new labels properly
plt.legend(title='Conditions', bbox_to_anchor=(1.05, 1), loc='upper left')

# Rotate x-axis labels for clarity
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
#%%

import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = [
    [0.36656891495601174, '8M', 'together', 'full_seq', '<cls>'],
    [0.36803519061583588, '8M', 'together', 'cdr', '<cls>'],
    [0.36070381231671556, '8M', 'together', 'full_seq', '_G'],
    [0.37536656891495680, '8M', 'together', 'cdr', '_G'],
    [0.35043988269794724, '8M', 'together', 'full_seq', '_GGGGS'],
    [0.36950146627565983, '8M', 'together', 'cdr', '_GGGGS'],
    [0.32991202346041054, '8M', 'separate', 'full_seq', '_GGGGS'],
    [0.35777126099706746, '8M', 'separate', 'cdr', '_GGGGS'],
    [0.36656891495601174, '35M', 'together', 'full_seq', '<cls>'],
    [0.37829912023460413, '35M', 'together', 'cdr', '<cls>'],
    [0.35043988269794724, '35M', 'together', 'full_seq', '_G'],
    [0.36803519061583583, '35M', 'together', 'cdr', '_G'],
    [0.36950146627565983, '35M', 'together', 'full_seq', '_GGGGS'],
    [0.38269794721407624, '35M', 'together', 'cdr', '_GGGGS'],
    [0.34457478005865105, '35M', 'separate', 'full_seq', '_GGGGS'],
    [0.38709677419354843, '35M', 'separate', 'cdr', '_GGGGS'],
    [0.35630498533724341, '150M', 'together', 'full_seq', '<cls>'],
    [0.37976539589442815, '150M', 'together', 'cdr', '<cls>'],
    [0.36363636363636365, '150M', 'together', 'full_seq', '_G'],
    [0.37096774193548391, '150M', 'together', 'cdr', '_G'],
    [0.34310850439882695, '150M', 'together', 'full_seq', '_GGGGS'],
    [0.37683284457478006, '150M', 'together', 'cdr', '_GGGGS'],
    [0.35190615835777131, '150M', 'separate', 'full_seq', '_GGGGS'],
    [0.36510263929618771, '150M', 'separate', 'cdr', '_GGGGS'],
    [0.3944281524926686, '3B', 'together', 'full_seq', '<cls>'],
    [0.41055718475073316, '3B', 'together', 'cdr', '<cls>'],
    [0.37976539589442815, '3B', 'separate', 'full_seq', '<cls>'],
    [0.40762463343108507, '3B', 'separate', 'cdr', '<cls>'],
    [0.28152492668621704, 'blum', '', 'full_seq', ''],
    [0.23900293255131966, 'blum', '', 'cdr', '']
]

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['Value', 'M', 'Type', 'SeqType', 'Label'])

# Set up plot
plt.figure(figsize=(10, 6))

# Define color for 'cdr' (red) and default for other types (blue)
cdr_color = 'red'
default_color = 'blue'

# Define markers for each 'Label'
marker_dict = {
    '_G': 'o',  # Circle marker for '_G'
    '_GGGGS': 's',  # Square marker for '_GGGGS'
    '<cls>': 'D',  # Diamond marker for '<cls>'
    '': 'X'  # Cross marker for 'blum'
}

# Plot each combination of SeqType, Type, and M
for (seq_type, typ) in df[['SeqType', 'Type']].drop_duplicates().values:
    subset = df[(df['SeqType'] == seq_type) & (df['Type'] == typ)]
    
    # Line style: solid for 'separate', dashed for 'together'
    linestyle = 'solid' if typ == 'separate' else 'dashed'
    
    for label in subset['Label'].unique():
        label_subset = subset[subset['Label'] == label]
        
        # Color the points by 'cdr' (red for 'cdr', blue for others)
        color = cdr_color if 'cdr' in label else default_color
        
        # Use different markers based on linker type
        marker = marker_dict.get(label, 'X')  # Default cross marker for unknown labels
        
        plt.plot(label_subset['M'], label_subset['Value'], 
                 label=f'{seq_type} - {typ} - {label}' if typ == 'together' else f'{seq_type} - {typ}', 
                 marker=marker, linestyle=linestyle, color=color)

# Customize plot
plt.title('Plot of Values by M Value')
plt.xlabel('M Values')
plt.ylabel('Value')

# Adjust legend to handle the new labels properly
plt.legend(title='Conditions', bbox_to_anchor=(1.05, 1), loc='upper left')

# Rotate x-axis labels for clarity
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
