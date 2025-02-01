import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df=pd.read_csv('MasterDF.csv')

print(df.columns.tolist())
'''
#### REORDER COLUMN#####

desired_columns = [
    'Year', 'Country_x', 'Continent','Code', 'Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)',
    'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 'Alcohol use disorders (%)',
    'Average Mental Heatlh Rate %', 'Mood Disorder %', 'Compulsion Disorder %',
    'Pop. w Mood Disorder', 'Pop. w Schizophrenia', 'Pop. w Depression', 'Pop. w Drug Dependency',
    'Pop. w Alcohol Dependency', 'Pop. w Bipolar', 'Pop. w Eating Disorder', 'Pop. w Anxiety',
    'agricultural_land%', 'forest_land%', 'land_area', 'avg_precipitation', 'trade_in_services%',
    'control_of_corruption_estimate', 'control_of_corruption_std', 'access_to_electricity%',
    'renewvable_energy_consumption%', 'CO2_emisions', 'other_greenhouse_emisions', 'population_density',
    'inflation_annual%', 'goverment_effectiveness_estimate', 'goverment_effectiveness_std',
    'individuals_using_internet%', 'military_expenditure%', 'GDP_current_US', 'political_stability_estimate',
    'political_stability_std', 'rule_of_law_estimate', 'rule_of_law_std', 'regulatory_quality_estimate',
    'regulatory_quality_std', 'government_expenditure_on_education%', 'government_health_expenditure%',
    'birth_rate', 'death_rate', 'life_expectancy_at_birth', 'population', 'rural_population',
    'voice_and_accountability_estimate', 'voice_and_accountability_std', 'GDP per Capita',
    'Healthcare Expenditure (U$D)', 'Education Expenditure (U$D)', 'Mental Health Market Size (U$D)'
]

# Reorder columns
df = df[desired_columns]
df.to_csv('MasterDF_v2.csv', index=False)
print(df.dtypes)'''

'''######################  START HERE ######################'''
df=pd.read_csv('MasterDF_v2.csv')
numeric_df = df.select_dtypes(include='number')

########    VISUALS 

plt.figure(figsize=(23,23))
sns.set(font_scale=1.2)
sns.heatmap(numeric_df.corr(), cmap='coolwarm')
plt.title('Heatmap of DataFrame') # Rotate x-axis labels by 45 degrees
plt.show()

'''######################## RANDOM FOREST PERMUTATION #############################'''
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
df.dropna(inplace=True)

######################## For Average Mental Illness ############################
features = ['Average Mental Heatlh Rate %','agricultural_land%', 'forest_land%', 'land_area', 'avg_precipitation', 'trade_in_services%', 'control_of_corruption_estimate', 'control_of_corruption_std', 'access_to_electricity%', 'renewvable_energy_consumption%', 'CO2_emisions', 'other_greenhouse_emisions', 'population_density', 'inflation_annual%', 'goverment_effectiveness_estimate', 'goverment_effectiveness_std', 'individuals_using_internet%', 'military_expenditure%', 'GDP_current_US', 'political_stability_estimate', 'political_stability_std', 'rule_of_law_estimate', 'rule_of_law_std', 'regulatory_quality_estimate', 'regulatory_quality_std', 'government_expenditure_on_education%', 'government_health_expenditure%', 'birth_rate', 'death_rate', 'life_expectancy_at_birth', 'population', 'rural_population','voice_and_accountability_std', 'GDP per Capita', 'Healthcare Expenditure (U$D)', 'Education Expenditure (U$D)', 'Mental Health Market Size (U$D)']

######################################## Train/test split #########################################

df_train, df_test = train_test_split(df, test_size=0.10)
df_train = df_train[features]
df_test = df_test[features]

X_train, y_train = df_train.drop('Average Mental Heatlh Rate %',axis=1), df_train['Average Mental Heatlh Rate %']
X_test, y_test = df_test.drop('Average Mental Heatlh Rate %',axis=1), df_test['Average Mental Heatlh Rate %']

################################################ Train #############################################

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

############################### Permutation feature importance #####################################
import rfpimp
imp = rfpimp.importances(rf, X_test, y_test)
y_pred = rf.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)


############################################## Plot ################################################

fig, ax = plt.subplots(figsize=(12,12))

# Sort the DataFrame by importance score in descending order
imp_sorted = imp.sort_values(by='Importance', ascending=False)

# Define colormap
cmap = plt.cm.get_cmap('coolwarm')

# Plot horizontal bar chart with gradient color
bars = ax.barh(imp_sorted.index, imp_sorted['Importance'], height=0.8, 
               color=cmap(imp_sorted['Importance']/imp_sorted['Importance'].max()))

# Add labels and title
ax.set_xlabel('Importance score')
ax.set_title('What Statistic Matters Most When Predicting Mental Illness %?', fontsize=18, weight='bold')

# Add a label for color scale
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
cbar.set_label('Color Scale')

# Invert y-axis
plt.gca().invert_yaxis()

# Add text annotation
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.2f}', va='center')

# Adjust layout
fig.tight_layout()

plt.show()
###########################################################################################

######################## For Axiety Disorder ############################
anxious = ['Anxiety disorders (%)','agricultural_land%', 'forest_land%', 'land_area', 'avg_precipitation', 'trade_in_services%', 'control_of_corruption_estimate', 'control_of_corruption_std', 'access_to_electricity%', 'renewvable_energy_consumption%', 'CO2_emisions', 'other_greenhouse_emisions', 'population_density', 'inflation_annual%', 'goverment_effectiveness_estimate', 'goverment_effectiveness_std', 'individuals_using_internet%', 'military_expenditure%', 'GDP_current_US', 'political_stability_estimate', 'political_stability_std', 'rule_of_law_estimate', 'rule_of_law_std', 'regulatory_quality_estimate', 'regulatory_quality_std', 'government_expenditure_on_education%', 'government_health_expenditure%', 'birth_rate', 'death_rate', 'life_expectancy_at_birth', 'population', 'rural_population','voice_and_accountability_std', 'GDP per Capita', 'Healthcare Expenditure (U$D)', 'Education Expenditure (U$D)', 'Mental Health Market Size (U$D)']

######################################## Train/test split #########################################

df_train, df_test = train_test_split(df, test_size=0.10)
df_train = df_train[anxious]
df_test = df_test[anxious]

X_train, y_train = df_train.drop('Anxiety disorders (%)',axis=1), df_train['Anxiety disorders (%)']
X_test, y_test = df_test.drop('Anxiety disorders (%)',axis=1), df_test['Anxiety disorders (%)']

################################################ Train #############################################

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

############################### Permutation feature importance #####################################
import rfpimp
imp = rfpimp.importances(rf, X_test, y_test)
y_pred = rf.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)


############################################## Plot ################################################

fig, ax = plt.subplots(figsize=(12,12))

# Sort the DataFrame by importance score in descending order
imp_sorted = imp.sort_values(by='Importance', ascending=False)

# Define colormap
cmap = plt.cm.get_cmap('coolwarm')

# Plot horizontal bar chart with gradient color
bars = ax.barh(imp_sorted.index, imp_sorted['Importance'], height=0.8, 
               color=cmap(imp_sorted['Importance']/imp_sorted['Importance'].max()))

# Add labels and title
ax.set_xlabel('Importance score')
ax.set_title('What Statistic Matters Most When Predicting Anxiety Disorder %?', fontsize=18, weight='bold')

# Add a label for color scale
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
cbar.set_label('Color Scale')

# Invert y-axis
plt.gca().invert_yaxis()

# Add text annotation
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.2f}', va='center')

# Adjust layout
fig.tight_layout()

plt.show()
######################## For Depression Disorder ############################
anxious = ['Depression (%)','agricultural_land%', 'forest_land%', 'land_area', 'avg_precipitation', 'trade_in_services%', 'control_of_corruption_estimate', 'control_of_corruption_std', 'access_to_electricity%', 'renewvable_energy_consumption%', 'CO2_emisions', 'other_greenhouse_emisions', 'population_density', 'inflation_annual%', 'goverment_effectiveness_estimate', 'goverment_effectiveness_std', 'individuals_using_internet%', 'military_expenditure%', 'GDP_current_US', 'political_stability_estimate', 'political_stability_std', 'rule_of_law_estimate', 'rule_of_law_std', 'regulatory_quality_estimate', 'regulatory_quality_std', 'government_expenditure_on_education%', 'government_health_expenditure%', 'birth_rate', 'death_rate', 'life_expectancy_at_birth', 'population', 'rural_population','voice_and_accountability_std', 'GDP per Capita', 'Healthcare Expenditure (U$D)', 'Education Expenditure (U$D)', 'Mental Health Market Size (U$D)']

######################################## Train/test split #########################################

df_train, df_test = train_test_split(df, test_size=0.10)
df_train = df_train[anxious]
df_test = df_test[anxious]

X_train, y_train = df_train.drop('Depression (%)',axis=1), df_train['Depression (%)']
X_test, y_test = df_test.drop('Depression (%)',axis=1), df_test['Depression (%)']

################################################ Train #############################################

rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)

############################### Permutation feature importance #####################################
import rfpimp
imp = rfpimp.importances(rf, X_test, y_test)
y_pred = rf.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)

print("R2 Score:", r2)
print("Mean Absolute Error:", mae)


############################################## Plot ################################################

fig, ax = plt.subplots(figsize=(12,12))

# Sort the DataFrame by importance score in descending order
imp_sorted = imp.sort_values(by='Importance', ascending=False)

# Define colormap
cmap = plt.cm.get_cmap('coolwarm')

# Plot horizontal bar chart with gradient color
bars = ax.barh(imp_sorted.index, imp_sorted['Importance'], height=0.8, 
               color=cmap(imp_sorted['Importance']/imp_sorted['Importance'].max()))

# Add labels and title
ax.set_xlabel('Importance score')
ax.set_title('What Factor Matters Most When Predicting Depression %?', fontsize=18, weight='bold')

# Add a label for color scale
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap=cmap), ax=ax)
cbar.set_label('Color Scale')

# Invert y-axis
plt.gca().invert_yaxis()

# Add text annotation
for bar in bars:
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.2f}', va='center')

# Adjust layout
fig.tight_layout()

plt.show()

''''################################## CLUSTER #################################'''

#Load the required packages
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df_2017 = df[df['Year'] == 2017]
df_2017.dropna(inplace=True)
df2 = df_2017[['Average Mental Heatlh Rate %', 'Anxiety disorders (%)',  'Depression (%)', 'government_health_expenditure%','population_density','GDP per Capita','life_expectancy_at_birth']]

# Data scaling
scaler=MinMaxScaler() #initialize
scaler.fit(df2)
scaled_df=scaler.transform(df2)

# Random Guess 4 clusters
km4= KMeans(n_clusters=4)
km4.fit(scaled_df)

## Elbow Mdethod and Siloutetter score
wvc=[]  ## Within clustre variation
sil_scores=[]


for i in range (2,15):
    km= KMeans(n_clusters=i)
    km.fit(scaled_df)
    wvc.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_df, km.labels_))
    
## Elbow Plot
 
plt.plot(range(2,15), wvc)
plt.xlabel('Num clusters')
plt.ylabel('Within cluster variation') # Plot show best K= 3 cluster

## Sil Score
plt.plot(range(2,15), sil_scores)
plt.xlabel('Num clusters')
plt.ylabel('Score') # Best Score at K= 3 cluster as well

# 5 CLUSTER ACCORDING TO PLOTS
km3= KMeans(n_clusters=3)
km3.fit(scaled_df)

## Elbow Mdethod and Siloutetter score
wvc=[]  ## Within clustre variation
sil_scores=[]

df_2017.dropna(inplace=True)
df_2017['label']=km3.labels_

df.head()
'''
# Define a dictionary mapping label values to player tiers
tier_mapping = {0: 'Superstars', 1: 'Bench', 2: 'Role/Key players'}

# Create the 'Player Tier' column by mapping label values to player tiers
df['Player Tier'] = df['label'].map(tier_mapping)'''


# Scatterplot of data. Interpret graph


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(13, 13))
sns.scatterplot(data=df_2017, x='population_density', y='Anxiety disorders (%)', hue='label', size='Average Mental Heatlh Rate %', style='label',palette='dark', sizes=(2, 850))

plt.xlabel('GDP Per Capita %')
plt.ylabel('Mental Health Rate %')
plt.legend(title='Tier Label', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.show()


plt.figure(figsize=(13, 13))
sns.scatterplot(data=df_2017, x='government_health_expenditure%', y='Average Mental Heatlh Rate %', hue='label', size='Average Mental Heatlh Rate %', style='label',palette='dark', sizes=(2, 850))

plt.xlabel('Healthcare Expenditure %')
plt.ylabel('Mental Health Rate %')
plt.legend(title='Tier Label', bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.show()

'''############# 3D LINEAR MODEL ##################'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D


X = df_2017[['life_expectancy_at_birth', 'avg_precipitation']].values.reshape(-1,2)
Y = df_2017['Anxiety disorders (%)']

######################## Prepare model data point for visualization ###############################

x = X[:, 0]
y = X[:, 1]
z = Y


# For x-axis range (Offensive Estimate)
life_range = df_2017['life_expectancy_at_birth'].min(), df_2017['life_expectancy_at_birth'].max()

# For y-axis range (Defensive Estimate)
pop_range = df_2017['avg_precipitation'].min(), df_2017['avg_precipitation'].max()

# For z-axis range (Salary)
MH_range = df_2017['Anxiety disorders (%)'].min(), df_2017['Anxiety disorders (%)'].max()


# Adjusted ranges
x_min, x_max = life_range
y_min, y_max = pop_range

# Use salary range for z-axis
z_min, z_max = MH_range

# Generate prediction grid using adjusted ranges
x_pred = np.linspace(x_min, x_max, 30)
y_pred = np.linspace(y_min, y_max, 30)
xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

################################################ Train #############################################
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import linear_model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# Train the linear regression model on the training data
ols = linear_model.LinearRegression()
model = ols.fit(X_train, y_train)
predicted = model.predict(model_viz)
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)

r2 = r2_score(y_test, y_pred)
print("R²:", r2)
############################################## Plot ################################################

#######################
###
fig = plt.figure(figsize=(15, 12))  
ax1 = fig.add_subplot(111, projection='3d', aspect='auto')

# Scale the size of each plot based on the 'Anxiety disorder(%)' column
size_scale = 40  # Adjust this value as needed
sizes = df_2017['Anxiety disorders (%)'] * size_scale

# Plot the scatter plot with adjusted size
scatter = ax1.scatter(x, y, z, c=df_2017['label'], cmap='viridis', zorder=15, marker='o', s=sizes, alpha=1)
ax1.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=25, edgecolor='black')

ax1.set_xlabel('Life Expectancy', fontsize=20)
ax1.set_ylabel('Avg Pertipitation', fontsize=20)
ax1.set_zlabel('Anxiety %', fontsize=20)  # Increase font size here

ax1.locator_params(nbins=4, axis='x')
ax1.locator_params(nbins=5, axis='x')

ax1.view_init(elev=12, azim=250)

fig.suptitle('$RMSE = %.2f$' % rmse, fontsize=50, fontweight='bold')

fig.colorbar(scatter, label='Cluster Group')  # Add colorbar

fig.tight_layout()
plt.show()

#### Convert GIF

for ii in np.arange(0, 360, 1):
    ax1.view_init(elev=12, azim=ii)
    fig.savefig('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/Visuals/3d Mental/gif_image%d.png' % ii)

    
# Close the figure
plt.close(fig)

###### CONVERT CLUSTER DF TO CSV
df_2017.to_csv('ClusterMentalHealth_2017.csv', index=False)
