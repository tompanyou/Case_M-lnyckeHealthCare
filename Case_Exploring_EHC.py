#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# # Load in csv files as a dictionary 

# In[17]:


import os
import pandas as pd

folder_path = 'csv'

# Get a list of all csv files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

# Create an empty dictionary to store the dataframes
data_dict = {}

# Loop through each csv file and load it into a dataframe
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    data_dict[file] = df

# Now the data_dict contains all the dataframes, with the key being the filename


# In[430]:


patients_df = data_dict["patients.csv"]


# # Choose single patient and load their data into a separate dictionary

# In[431]:


conditions_df = data_dict["conditions.csv"]
single_patient_name = conditions_df.PATIENT[1] 


# In[432]:


single_patient_dict = {}

for name, df in data_dict.items():
    try:
        single_patient_dict[name] = df[df.PATIENT==single_patient_name]
    except:
        print(name, "has no such column")


# # For all dataframes with a description column, count the number of occurences and plot up as a bar chart

# In[433]:


df = single_patient_dict["conditions.csv"]


# In[434]:


for name, df in single_patient_dict.items():
    try:
        print(name)
        #temp = df[["START", "STOP", "DESCRIPTION"]]
        #display(temp.reset_index(drop=True))
#         sns.countplot(x='DESCRIPTION', data=df)
        
#         plt.xticks(rotation=45)
#         plt.show()
        counts = df['DESCRIPTION'].value_counts()
        plt.pie(counts, labels=counts.index)
        #Add text labels for the counts
#         for i, count in enumerate(counts):
#             plt.text(1.2, 1.1-i*0.2, f'{counts.index[i]}: {count}', color='black', fontsize=20)
        #plt.savefig(name+".png")
        plt.show()
        
    except:
        print(name, "has no description column")
        display(df)
    #display(df)


# In[510]:


encounters_df = single_patient_dict["encounters.csv"].copy()


# In[511]:


encounters_df['START'] = pd.to_datetime(encounters_df['START'])
sum_by_year_encounter = encounters_df.groupby(encounters_df['START'].dt.year)['TOTAL_CLAIM_COST'].sum()


# In[512]:


import pandas as pd


# Create a new index with all years
all_years = pd.RangeIndex(start=sum_by_year_medication.index.min(), stop=sum_by_year_medication.index.max()+1)

# Reindex the series with the complete index and fill missing values with zero
s = sum_by_year_medication.reindex(all_years, fill_value=0)

print(s)


# In[513]:


import pandas as pd


# Create a new index with all years
all_years = pd.RangeIndex(start=sum_by_year_encounter.index.min(), stop=sum_by_year_encounter.index.max()+1)

# Reindex the series with the complete index and fill missing values with zero
t = sum_by_year_encounter.reindex(all_years, fill_value=0)


# In[514]:


all_years = pd.RangeIndex(start=sum_by_year.index.min()-1, stop=sum_by_year.index.max()+1)

pro = sum_by_year.reindex(all_years, fill_value=0)


# In[515]:


import pandas as pd
import matplotlib.pyplot as plt


plt.bar(pro.index, pro, label="Procedures")
plt.bar(s.index, s, bottom=pro,label="Medications")
plt.bar(t.index, t, bottom=s+pro, label="Encounters")
plt.legend()
plt.ylabel("Cost")
plt.xlabel("Year")
plt.title("Cost/Year for Patient")
plt.savefig("cost_patient")
plt.show()


# # Group the DESCRIPTION by year and count the number of occurrences and show as a table

# In[516]:


for name, df in single_patient_dict.items():
    try:
        temp = df.copy()
        temp["START"] = pd.to_datetime(temp['START'])
        temp['year'] = temp["START"].dt.year
        grouped_df = temp.groupby(['year', 'DESCRIPTION']).size().reset_index(name='count')
        pivoted_df = grouped_df.pivot(index='year', columns='DESCRIPTION', values='count')
        pivoted_df = pivoted_df.fillna(0)
        table = pivoted_df.style.format('{:.1f}')
        # Display the table in Jupyter notebook
        display(table)
        # sns.heatmap(pivoted_df, cmap='Blues')
        # plt.xlabel('DESCRIPTION')
        # plt.ylabel('Year')
        # plt.title('Count of Examinations by Year')
        # plt.show()
    except:
        print(name, "has no description column")
        display(df)
        
    #display(df)


# In[517]:


# Plot cost of medications


# In[518]:


medications_df


# In[519]:


sum_by_year_medication = medications_df.groupby(medications_df['START'].dt.year)['TOTALCOST'].sum()


# In[520]:


medications_df = single_patient_dict["medications.csv"].copy()
medications_df["START"] = pd.to_datetime(medications_df["START"])
plt.bar(sum_by_year_medication.index, sum_by_year_medication)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Total Cost')
plt.title('Cost/Year for medication')
# Show the plot
plt.savefig("medication_cost.png")


# # Plot base cost of procedures

# In[521]:


procedures_df


# In[522]:


sum_by_year = procedures_df.groupby(procedures_df['DATE'].dt.year)['BASE_COST'].sum()

# Print the result
print(sum_by_year)


# In[523]:


procedures_df = single_patient_dict["procedures.csv"].copy()
procedures_df["DATE"] = pd.to_datetime(procedures_df["DATE"])
# Create a bar chart
plt.bar(sum_by_year.index,sum_by_year)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Base Cost')
plt.title('Cost/Year for procedures')
# Show the plot
#plt.show()
plt.savefig("cost_procedures.png")


# In[524]:


procedures_df = single_patient_dict["procedures.csv"].copy()
procedures_df["DATE"] = pd.to_datetime(procedures_df["DATE"])
# Create a bar chart
plt.bar(procedures_df["DATE"].dt.year, procedures_df["BASE_COST"])

# Add labels and title>
plt.xlabel('Year')
plt.ylabel('Base Cost')
plt.title('Base Cost per Year')
# Show the plot
plt.show()


# # Showing table of the files with DATE and DESCRIPTION

# In[525]:


for name, df in single_patient_dict.items():
    try:
        temp = df.copy()
        temp["DATE"] = pd.to_datetime(temp['DATE'])
        temp['year'] = temp["DATE"].dt.year
        grouped_df = temp.groupby(['year', 'DESCRIPTION']).size().reset_index(name='count')
        pivoted_df = grouped_df.pivot(index='year', columns='DESCRIPTION', values='count')
        pivoted_df = pivoted_df.fillna(0)
        table = pivoted_df.style.format('{:.1f}')
        # Display the table in Jupyter notebook
        display(table)
        # sns.heatmap(pivoted_df, cmap='Blues')
        # plt.xlabel('DESCRIPTION')
        # plt.ylabel('Year')
        # plt.title('Count of Examinations by Year')
        # plt.show()
    except:
        print(name, "has no description column")
        #display(df)
        
    #display(df)


# In[531]:


observations_df = single_patient_dict["observations.csv"].copy()
observations_df = observations_df.drop(observations_df[observations_df.VALUE=="Never smoker"].index)
observations_df["VALUE"] = pd.to_numeric(observations_df["VALUE"])
observations_df['DATE'] = pd.to_datetime(observations_df['DATE'])
pivot_df = observations_df.pivot(index='DATE', columns='DESCRIPTION', values='VALUE')

# Plot the pivoted dataframe
pivot_df.plot()
df = pivot_df.copy()
# Add axis labels and a title
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.title('Values Over Time for Each Category')

# Show the plot
plt.show()


# In[534]:


x = df.columns[[0,1,2,5,25,26]]


# In[535]:


# Define the number of columns in the figure
num_cols = 3

# Create the figure and subplots
fig, axs = plt.subplots(nrows=len(x)//num_cols, ncols=num_cols, figsize=(15, 20))

df = pivot_df.copy()
# Plot each column in a separate subplot
for i, column in enumerate(x):
    
    row = i // num_cols
    col = i % num_cols
    temp = df[column].dropna()
    axs[row, col].plot(temp.index.year, temp.values)
    
    if len(column) > 25:
        axs[row, col].set_title(column, fontsize = 8)
    else:
        axs[row, col].set_title(column, fontsize = 20)
        
    
# Remove any unused subplots
for i in range(len(df.columns), axs.size):
    fig.delaxes(axs.flatten()[i])

# Show the resulting figure
plt.tight_layout()

#plt.show()
fig.savefig('my_plot2.png')


# # Exploring patients with the same conditions

# In[536]:


temp


# In[537]:


counts_conditions = conditions_df.DESCRIPTION.value_counts()


#counts_conditions[0:3].plot(kind='bar')
# set the labels and title
temp = counts_conditions[0:5]
ax = temp.plot(kind='bar')

# set the labels and title
#ax.set_xlabel('Category')
ax.set_ylabel('Count')
ax.set_title('Ocurrences of conditions')

# set the font size of the x-tick labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=80,fontsize=12)
plt.savefig("common_conditions.png")
# show the plot
plt.show()


# # How the conditions are treated

# In[538]:


careplan_df = data_dict["careplans.csv"]


# In[539]:


top_3_conditions = temp.index


# In[540]:


for condition in top_3_conditions:
    display(careplan_df[careplan_df.REASONDESCRIPTION == condition])
    print(np.unique(careplan_df[careplan_df.REASONDESCRIPTION == condition].DESCRIPTION))


# In[541]:


for name, df in data_dict.items():
    try:
        print(name)
        display(df)
        temp = df[df.REASONDESCRIPTION.isin(top_3_conditions)]
        #display(temp)
        cross_tab = pd.crosstab(temp['DESCRIPTION'], temp['REASONDESCRIPTION'])

        # display the crosstab table
        display(cross_tab)
        sns.countplot(x='REASONDESCRIPTION', data=df)
        plt.xticks(rotation=45)
        #plt.show()
    except:
        #print("")
        print(name, "has no description column")
        display(df)


# # Other common pattern characteristics for the three conditions

# In[542]:


encounters_df = data_dict["encounters.csv"]
encounters_top_3 = encounters_df[encounters_df.REASONDESCRIPTION.isin(top_3_conditions)]
encounters_top_3
#print(encounters_df["ENCOUNTERCLASS"].unique())


# In[547]:


imaging_df = data_dict["imaging_studies.csv"]


# In[548]:


imaging_df.BODYSITE_DESCRIPTION.unique()


# In[549]:


patients = encounters_top_3.PATIENT


# In[550]:


patients_top_3 = conditions_df[conditions_df.DESCRIPTION.isin(top_3_conditions)].PATIENT


# In[551]:


allergies_df = data_dict["allergies.csv"]
allergies_df


# In[552]:


immunization_df = data_dict["immunizations.csv"]
immunization_df


# In[553]:


for condition in top_3_conditions:
    patients = conditions_df[conditions_df.DESCRIPTION == condition].PATIENT
    allergies = allergies_df[allergies_df.PATIENT.isin(patients)]
    immunization = immunization_df[immunization_df.PATIENT.isin(patients)]
    print("\n")
    print(condition, "\n")
    print("Allergies")
    print(allergies.DESCRIPTION.unique())
    print("Immunization")
    print(immunization.DESCRIPTION.unique())


# In[554]:


np.unique(observations_df.VALUE)


# In[555]:


observations_df[observations_df.VALUE == "Never smoker"]


# In[556]:


observations_df


# In[557]:


conditions_df["duration"] = pd.to_datetime(conditions_df["STOP"])-pd.to_datetime(conditions_df["START"])
conditions_df


# In[558]:


conditions_df.groupby("DESCRIPTION")["duration"].count()


# In[559]:


conditions_df.groupby("DESCRIPTION")["duration"].sum()


# In[561]:


conditions_df


# In[562]:


conditions_df["START"] = pd.to_datetime(conditions_df["START"])
conditions_df["START"].dt.month


# In[563]:


import calendar
month_names = [calendar.month_name[i] for i in range(1, 13)]
month_names = [calendar.month_name[i] for i in range(1, 13)]


# In[564]:


for c in top_3_conditions:
    temp = conditions_df[conditions_df.DESCRIPTION == c]
    months = temp["START"].dt.month
    plt.hist(months)
    plt.title(c[:-11] + ": Seasonal outbreak")
    plt.xticks(range(1, 13), month_names, rotation=45)
    plt.tight_layout()
    
    plt.savefig(c+".png")
    plt.show()


# In[565]:


conditions_df["START"]


# In[566]:


top_3_conditions


# In[567]:


count = conditions_df.groupby("DESCRIPTION")["duration"].count()
count[top_3_conditions]


# In[568]:


avg_duration = conditions_df.groupby("DESCRIPTION")["duration"].sum()/conditions_df.groupby("DESCRIPTION")["duration"].count()
avg_duration[top_3_conditions]


# In[569]:


np.unique(observations_df[observations_df.DESCRIPTION == "Tobacco smoking status NHIS"].VALUE)


# In[570]:


observations_df.DESCRIPTION.unique()


# In[571]:


conditions_df[conditions_df.PATIENT == "00185faa-2760-4218-9bf5-db301acf8274"]


# In[572]:


categorical_values_index


# In[573]:


observations_df = data_dict["observations.csv"].copy()
# categorical_values_index = observations_df[observations_df.TYPE=="text"].index
# observations_df = observations_df.drop(observations_df[observations_df.TYPE=="text"].index)
# observations_df["VALUE"] = pd.to_numeric(observations_df["VALUE"])
# observations_agg = observations_df.groupby(['PATIENT', 'DESCRIPTION'])['VALUE'].mean().reset_index()

# observations_pivot = observations_agg.pivot(index='PATIENT', columns='DESCRIPTION', values='VALUE')
# observations_pivot.fillna(0, inplace=True)
# #display(observations_pivot)

# observations_pivot["Leukocytes [#/volume] in Blood by Automated count"]


#display(observations_df[observations_df.PATIENT == "00185faa-2760-4218-9bf5-db301acf8274"].drop_duplicates(subset=['PATIENT', 'DESCRIPTION']))
#observations_agg = observations_df.groupby(['PATIENT', 'DESCRIPTION'])['VALUE'].mean().reset_index()
#display(observations_df[observations_df.PATIENT == "00185faa-2760-4218-9bf5-db301acf8274"])
# observations_df.drop_duplicates(subset=['PATIENT', 'DESCRIPTION'], inplace=True)
# #observations_df = observations_df.drop(observations_df[observations_df.DESCRIPTION=="Tobacco smoking status NHIS"].index)
# #observations_df = observations_df.drop(observations_df[observations_df.VALUE=="Never smoker"].index)
# #observations_df["VALUE"] = pd.to_numeric(observations_df["VALUE"])
# #observations_df['DATE'] = pd.to_datetime(observations_df['DATE'])
# pivot_df = observations_df.pivot(index='PATIENT', columns='DESCRIPTION', values='VALUE')
# pivot_df


# In[574]:


np.unique(observations_df[observations_df.TYPE=="text"].VALUE)


# In[575]:


patients_top_3 = conditions_df[conditions_df.DESCRIPTION.isin(top_3_conditions)]
patients_top_3 = patients_top_3[["PATIENT", "DESCRIPTION"]].reset_index(drop=True)
patients_top_3 = patients_top_3.drop_duplicates(subset="PATIENT")
patients_top_3


# In[576]:


observations_df


# In[577]:


len(patients_top_3.PATIENT)


# In[ ]:


k = 0
nbr_former_smokers = 0
nbr_current_smoker = 0
nbr_never_smoker = 0
for p in patients_top_3.PATIENT:
    temp = observations_df[(observations_df.PATIENT==p) & (observations_df.DESCRIPTION == "Tobacco smoking status NHIS")]
    if "Current every day smoker" in temp.values:
        nbr_current_smoker += 1
    elif "Former smoker" in temp.values:
        nbr_former_smokers += 1
    elif "Never smoker" in temp.values:
        nbr_never_smoker += 1

print(nbr_current_smoker, nbr_former_smokers, nbr_never_smoker)

    


# In[ ]:


conditions_df[conditions_df.DESCRIPTION == condition].PATIENT


# In[ ]:


factors = ['Respiratory rate', 'Pain severity - 0-10 verbal numeric rating [Score] - Reported', 'Body temperature', 'Leukocytes [#/volume] in Blood by Automated count']
#regular.columns[1:10]
for column in factors:
    for condition in top_3_conditions:
        patients  = conditions_df[conditions_df.DESCRIPTION == condition].PATIENT
        a = regular[regular.PATIENT.isin(patients)][column]
        plt.hist(a)
        plt.title(condition + " " + column,)
        plt.show()

