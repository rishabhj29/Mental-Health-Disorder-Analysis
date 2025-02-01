import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

################# CLEANING STEPS, PLEASE KEEP IN QUOTES #################################################################################
'''

mental= pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/Mental Health(1999-2019).csv')

econ= pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/World Economy (1999-2019).csv')

emp = pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/World Unemployment (1999-2019).csv')

details= pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/World Unemployment Detailed (2014-2024).csv')

US_mh = mental[mental['Country'] == 'United States']

commor= pd.read_csv('/Volumes/Untitled/Users/dino/Desktop/MSBA/Spring 24/Modeling/Visuals/MentalHealth_VizProject/Archive/USA Comobidities (1999-2019)_v2.csv')


econ_sub= econ


# Assuming 'econ' is your DataFrame containing economic data

# List all columns
econ_columns = econ.columns
print(econ_columns)

# Assuming 'mental' is your DataFrame containing mental health data

################################################################
"GROUPING MENTAL HEALTH TOGETHER AND MAKE DIMENSION REDUCTIONS"


# List of mental health condition columns
mental_health_conditions = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)', 
                            'Anxiety disorders (%)', 'Drug use disorders (%)', 'Depression (%)', 
                            'Alcohol use disorders (%)']
mental['Average Mental Heatlh Rate %'] = mental[mental_health_conditions].mean(axis=1)
# Create a new column 'Average Mental Health Rate' containing the average rate across all mental health conditions


# Delete rows where 'Country' is 'World'
mental = mental[mental['Country'] != 'World']

# Grouping diseases together

#Emotional Disorder
mood=['Bipolar disorder (%)','Depression (%)']
mental['Mood Disorder %'] = mental[mood].sum(axis=1)

#Compulsive Disorder
compulsion = ['Alcohol use disorders (%)','Drug use disorders (%)','Eating disorders (%)' ]
mental['Compulsion Disorder %'] = mental[compulsion].sum(axis=1)
##################################################################
# Extracting unique countries from the 'Country' column
unique_countries = mental['Country'].unique()

# Printing the list of unique countries
for country in unique_countries:
    print(country)


### REMOVING NON COUNTRIES ###################################################

non_countries = [
    'Andean Latin America', 'Australasia', 'Caribbean', 'Central Asia', 'Central Europe',
    'Central Europe, Eastern Europe, and Central Asia', 'Central Latin America', 'Central Sub-Saharan Africa',
    'East Asia', 'Eastern Europe', 'Eastern Sub-Saharan Africa', 'High SDI', 'High-income',
    'High-income Asia Pacific', 'High-middle SDI', 'Latin America and Caribbean', 'Low SDI',
    'Low-middle SDI', 'Micronesia (country)', 'Middle SDI', 'North Africa and Middle East',
    'North America', 'Northern Ireland', 'Oceania', 'South Asia', 'Southeast Asia',
    'Southeast Asia, East Asia, and Oceania', 'Southern Latin America', 'Southern Sub-Saharan Africa',
    'Sub-Saharan Africa', 'Tropical Latin America', 'Western Europe', 'Western Sub-Saharan Africa'
]

# Filter out 'World' and non-country entries from the 'Country' column
mental2 = mental[~mental['Country'].isin(['World'] + non_countries)]
mental2.to_excel('mental_data_v3.xlsx', index=False)
###################################################
""" CREATING ID"""
#Creating ID 

# Create the "ID" column by concatenating "Country" and "Year"
## Mental Health Data
mental2['Year Actual'] = mental2['Year'].str.extract(r'(\d{4})')
mental2['ID'] = mental2['Country'] + mental2['Year Actual'].astype(str)

econ_sub['Year Actual'] = econ_sub['Year'].str.extract(r'(\d{4})')
econ_sub['ID'] = econ_sub['Country'] + econ_sub['Year Actual'].astype(str)

merged= pd.merge(mental2, econ_sub, on='ID', how='inner')

exact_matches = (merged['Country_x'] == merged['Country_y']).sum()

###################################################

""" CHECKING IF THE IDS MATCH THE COUNTRIES AND YEARS"""
# Initialize a list to store the row numbers of exact matches
exact_match_rows = []

# Iterate over the DataFrame and find exact matches
for index, row in merged.iterrows():
    if row['Country_x'] == row['Country_y']:
        exact_match_rows.append(index)

# Calculate the percentage of exact matches
exact_matches_count = len(exact_match_rows)
total_rows = len(merged)
percentage_exact_matches = (exact_matches_count / total_rows) * 100

print(f"The percentage of exact matches between 'Country_x' and 'Country_y' is: {percentage_exact_matches:.2f}%")

# Delete columns 'Country_Y' and 'Year_Y'
merged.drop(columns=['Country_y', 'Year_y','Year Actual_y',], inplace=True)
merged['GDP per Capita'] = merged['GDP_current_US'] / merged['population']
merged.to_excel('merge_data_v7.xlsx', index=False)


df= pd.read_excel('merge_data_v7.xlsx')
df['Year'] = df['Year_x'].str.extract(r'(\d{4})')
df.drop(columns=['Year_x'], inplace=True)
#Drop Columns with more than 50% Null
null_percentage = (df.isnull().sum() / len(df)) * 100
print(null_percentage)
threshold = 0.7 * len(df)
df = df.dropna(thresh=threshold, axis=1)

#Convert
df.to_excel('merge_data_v8.xlsx', index=False)
'''
 ############################## START HERE ################################################################

df=pd.read_excel('merge_data_v8.xlsx')
print(df.columns)

'''CREATING COLUMNS FOR POPULATION W MENTAL HEALTH'''

df['Pop. w Mood Disorder'] = df['Mood Disorder %'] * df['population']/100
df['Pop. w Schizophrenia'] = df['Schizophrenia (%)'] * df['population']/100
df['Pop. w Depression'] = df['Depression (%)'] * df['population']/100
df['Pop. w Drug Dependency'] = df['Drug use disorders (%)'] * df['population']/100
df['Pop. w Alcohol Dependency'] = df['Alcohol use disorders (%)'] * df['population']/100
df['Pop. w Bipolar'] = df['Bipolar disorder (%)'] * df['population']/100
df['Pop. w Eating Disorder'] = df['Eating disorders (%)'] * df['population']/100
df['Pop. w Anxiety'] = df['Anxiety disorders (%)'] * df['population']/100


'''Calculating Mental Health Expenditure By Estimating 2% Health Care Expenditure'''

### HEALTHCARE/ EDUCATION / MENTAL HEALTH EXPENDITURE AS CALCULATION OF GDP

df['Healthcare Expenditure (U$D)'] = df['GDP_current_US'] * df['government_health_expenditure%']/100
df['Education Expenditure (U$D)'] = df['GDP_current_US'] * df['government_expenditure_on_education%']/100

# According to World Bank, Countries spend on average no more than 2% of healtcare expenditure on mental health
# Source: https://blogs.worldbank.org/en/health/mental-health-conditions-rise-countries-must-prioritize-investments#:~:text=Yet%2C%20countries%20allocate%20on%20average,an%20outdated%20approach%3A%20psychiatric%20hospitals.

# Use Healthcare Expenditure * 2% = Market Size for Mental Healthcare
df['Mental Health Market Size (U$D)'] = df['Healthcare Expenditure (U$D)'] * 0.02
#####################################################################

df.to_excel('merge_data_v10.xlsx', index=False)
print(df.columns)

df=pd.read_excel('merge_data_v10.xlsx')

######### Create Continent Label #######################
# Define lists for each continent
asia = ['Afghanistan', 'Armenia', 'Myanmar','Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei', 'Burma (Myanmar)', 'Cambodia', 'China', 'East Timor', 'Georgia', 'India', 'Indonesia', 'Iran', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'North Korea', 'South Korea', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Nepal', 'Oman', 'Pakistan', 'Philippines', 'Qatar', 'Russian Federation', 'Saudi Arabia', 'Singapore', 'Sri Lanka', 'Syria', 'Tajikistan', 'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']
europe = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine', 'United Kingdom', 'Vatican City']
africa = ['Algeria',"Cote d'Ivoire" ,'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Democratic Republic of the Congo', 'Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe']
oceania = ['Australia', 'American Samoa','Guam','Federated States of Micronesia', 'Northern Mariana Islands','Fiji', 'Kiribati', 'Marshall Islands', 'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu']
north_america = ['Antigua and Barbuda','Bermuda','Greenland', 'Puerto Rico','Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica', 'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica', 'Mexico', 'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Trinidad and Tobago', 'United States']
south_america = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela']

# Create a dictionary to map countries to their continents
country_to_continent = {}
for country in asia:
    country_to_continent[country] = 'Asia'
for country in europe:
    country_to_continent[country] = 'Europe'
for country in africa:
    country_to_continent[country] = 'Africa'
for country in oceania:
    country_to_continent[country] = 'Oceania'
for country in north_america:
    country_to_continent[country] = 'North America'
for country in south_america:
    country_to_continent[country] = 'South America'

# Create a new column "Continent" based on the country-to-continent mapping
df['Continent'] = df['Country_x'].map(country_to_continent)

countries_with_nan_continent = df[df['Continent'].isna()]['Country_x'].unique().tolist()
print(countries_with_nan_continent)

df.to_excel('master.xlsx', index=False)
df.to_csv('master.csv', index=False)


