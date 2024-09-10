#!/usr/bin/env python3

"""
@author: Ayobami Ogunmolasuyi
Thayer School of Engineering at Dartmouth College
ayobami.o.ogunmolasuyi.th@dartmouth.edu

FirnLearn: A Neural Network based approach to Firn Densification Modeling for Antarctica
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import ascii_letters
import pandas as pd
import seaborn as sns

SummedUp_df = pd.read_csv("training_df.csv")

SummedUp_df = SummedUp_df.drop(columns = ['Unnamed: 0','Profile','Citation','Method','Date','Timestamp','Start_Depth','Stop_Depth'])

%%plot heatmap
plt.figure(figsize=(16, 6))
# Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
heatmap = sns.heatmap(SummedUp_df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
