import numpy as np
import random
import torch
import pandas as pd
import os, sys
import time
sys.path.insert(0, os.path.abspath(".."))

from boostedCP.getData import get_real_data
from boostedCP.utils import local_preboost,rounding
from boostedCP.condcov_local_boost import condcov_local_boost

# Load data

data_name = "concrete"
seed = 2023
ratio = 70

print("seed ",seed)
print("ratio ",ratio)
print("data_name ",data_name)

os.environ['data_name'] = data_name
os.environ['seed'] = str(seed)
os.environ['ratio'] = str(ratio)
os.environ['print_tree'] = "FALSE"

miscov_rate = 0.1

params = dict()
params['data']   = data_name              # Name of dataset
params['ratio']  = ratio                  # Percentage of data used for training
params['seed']   = seed                   # Random seed

# Set random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Run experiment. This will take around 3 minutes.
## The Medical Expenditure Panel Survey (meps) data is subject to copyright and usage rules.
## Therefore, here we only reproduce Figure A2(c) for the concrete dataset.
# This script serves as an illustrative example. Due to runtime considerations, the maximum number of boosting rounds (n_rounds_cv) is set to 50, rather than 500.

start_time = time.time()

X_train, y_train, X_cal, y_cal, X_test, y_test= get_real_data(params)

cache_dir = "cache/"+data_name+"/seed_"+str(seed)+"_ratio_"+str(ratio)+"_contrast_rfun.json"

dictionary = condcov_local_boost(X_train, y_train, X_cal, y_cal,X_test, y_test, 
                      miscov_rate, seed,data_name,cache_dir,n_rounds_cv=50,learning_rate=1,n_folds = 2,verbose=False)
                      
local_rounded_condcov_group = [round((1 - x) * 100, 1) for x in dictionary["local_condcov_group"]]
boosted_rounded_condcov_group = [round((1 - x) * 100, 1) for x in dictionary["boosted_condcov_group"]]

print("\nResults:\n", dictionary)
print("\nBaseline Local Procedure:")
print("Miscoverage rates at each leaf node of the contrast tree (%): ", local_rounded_condcov_group)
print("Size of each leaf node of the contrast tree (%): ", rounding(dictionary["local_wt"]))
print("\nBoosted Procedure:")
print("Miscoverage rates at each leaf node of the contrast tree (%): ", boosted_rounded_condcov_group)
print("Size of each leaf node of the contrast tree (%): ", rounding(dictionary["boosted_wt"]))


runtime = time.time() - start_time
print("--- runtime: %s minues ---" % (runtime/60))

# Please refer to the output file 'Figure3.out' located in the same directory as this script for results.
