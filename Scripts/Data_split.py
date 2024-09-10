#!/usr/bin/env python3

"""
@author: Ayobami Ogunmolasuyi
Thayer School of Engineering at Dartmouth College
ayobami.o.ogunmolasuyi.th@dartmouth.edu

FirnLearn: A Neural Network based approach to Firn Densification Modeling for Antarctica

Creating the training, validation, and test datasets


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df_all = pd.read_csv("training_df.csv",index_col=['Profile'])

df_all2 = pd.read_csv("training_df.csv")

