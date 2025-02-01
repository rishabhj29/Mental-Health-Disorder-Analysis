# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:42:09 2024

@author: nangi
"""

#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

usa = pd.read_csv("USA.csv")

# Correlation matrix
correlation_matrix = usa.corr()
correlation_matrix
# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, cmap='magma', linewidths=.5)
plt.title('Correlation Heatmap of Variables for USA')
plt.show()

correlation_matrix.to_csv('CM1.csv')
