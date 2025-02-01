#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 01:00:17 2024

@author: edithkee
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Load data from MasterDF file
master_df = pd.read_excel('MasterDF.xlsx')


# Filter data for the United States
us_data = master_df[master_df['Country_x'] == 'United States']


# Data
years = us_data['Year']
anxiety_percentages = us_data['Anxiety disorders (%)']
CO2_emissions = us_data['CO2_emisions']



# Create a DataFrame
nightingale_df = pd.DataFrame({'Year': years, 'Anxiety_Disorder': anxiety_percentages, 'CO2_Emissions': CO2_emissions})

# Normalize the data for plotting
normalized_anxiety = (anxiety_percentages - anxiety_percentages.min()) / (anxiety_percentages.max() - anxiety_percentages.min())
normalized_CO2 = (CO2_emissions - CO2_emissions.min()) / (CO2_emissions.max() - CO2_emissions.min())


# Define a colormap with a unique color for each year
num_years = len(years)
colors = plt.cm.viridis(np.linspace(0, 1, num_years))

# Plot
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(polar=True))

# Plot CO2 emissions as bars extending outward from the center
theta = np.linspace(0, 2 * np.pi, num_years, endpoint=False)
width = 2 * np.pi / num_years

for i, (co2, theta_val, color) in enumerate(zip(normalized_CO2, theta, colors)):
    ax.bar(theta_val, co2, width=width, color=color, alpha=0.5, label=years.iloc[i])

# Plot anxiety disorder percentages as an orange line plot with thicker linewidth
ax.plot(theta, normalized_anxiety, color='orange', label='Anxiety Disorder (%)', linewidth=2)

# Add labels
ax.set_xticks(theta)
ax.set_xticklabels(years, fontsize=10)
ax.set_title('Nightingale Chart of CO2 Emissions and Anxiety Disorder (%) in the US by Year')


# Show the plot
plt.show()