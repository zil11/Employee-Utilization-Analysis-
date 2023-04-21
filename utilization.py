# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:19:37 2023

@author: jeffe
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\jeffe\Desktop\PITT\ECON 2814 Evidence-Based Analysis\Assignments\Assignment 1 - Rivers Agile\df.csv')

df.dtypes

# encode date and conver

df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

# add a new column for the month of the year

df['month'] = df['date'].dt.month

# add a new column for the week number
df['week'] = df['date'].dt.week

# add a new column for the day of the week
df['day'] = df['date'].dt.weekday + 1

df['date'] = df['date'].dt.date


# define a function that returns the billable hours of work
def Bhours(row):
    if row['billable'] == 1:
        return row['hrs']
    else:
        return 0

# apply the function to create a new column for the billable hours of work
df['Bhours'] = df.apply(Bhours, axis=1)

# use the value_counts() method to count the frequency

task_freq = df['task'].value_counts()
print(task_freq)

proj_freq = df['project'].value_counts()
print(proj_freq)

name_freq = df['name'].value_counts()
print(name_freq)

id_freq = df['id'].value_counts()
print(id_freq)

# plot the frequency counts
task_freq.plot(kind='bar')
plt.title('Frequency of Each Task Type')
plt.xlabel('Task Type')
plt.ylabel('Frequency')
plt.show()

proj_freq.plot(kind='bar')
plt.title('Frequency of Each Project')
plt.xlabel('Project Type')
plt.ylabel('Frequency')
plt.show()

name_freq.plot(kind='bar')
plt.title('Frequency of Each Name')
plt.xlabel('Name')
plt.ylabel('Frequency')
plt.show()

id_freq.plot(kind='bar')
plt.title('Frequency of Each ID')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.show()

sns.countplot(data = df, x = 'contractor')
plt.title('Number of Employee vs. Contractor')
plt.show()

# group the dataframe by task type and calculate the mean of billable
billables_by_task = df.groupby('task')['billable'].mean()
billables_by_task = pd.DataFrame([billables_by_task]).transpose()
billables_by_project = df.groupby('project')['billable'].mean()
billables_by_employ_type = df.groupby('contractor')['billable'].mean()

# plot the frequency counts and share of billable tasks
ax = billables_by_task.plot(kind='bar')
ax.set_title('Billable vs. Not Billable')
ax.set_xlabel('Task Type')
ax.set_ylabel('Share of Billable Hours')
plt.show()

hrs_task = df.groupby('task')['hrs'].sum()
hrs_task = pd.DataFrame([hrs_task]).transpose()

billables_by_task['hrs'] = hrs_task['hrs']

# group the data by task type and employment type and count the groups
hrs_by_task = df.groupby(['task', 'contractor'])['hrs'].count()

# group the counts by job_type and normalize each group
normalized = hrs_by_task.groupby('task').apply(lambda x: x / float(x.sum()))

# group the data by name, find utilization rate for each name within the timespan of the dataset  
util = df.groupby('name')['hrs', 'Bhours', 'contractor'].sum()
util['contractor'] = util['contractor'].apply(lambda x: 1 if x > 0 else x)
util['utilization'] = util['Bhours'] / util['hrs']

# reset the index to make name a regular column
util.index.name = 'name'
util = util.reset_index()

# create box plot
sns.boxplot(x='contractor', y='utilization', data=util)

# set plot labels and title
plt.xlabel('Contractor Status')
plt.ylabel('Average Utilization Rate')
plt.title('Average Utilization Rate by Employment Status')

# show plot
plt.show()

# group by contractor status and calculate the mean utilization rate
grouped_util = util.groupby('contractor')['utilization'].mean()

# create bar chart
grouped_util.plot(kind='bar')

# set plot labels and title
plt.xlabel('Contractor Status')
plt.ylabel('Average Utilization Rate')
plt.title('Average Utilization Rate by Employment Status')

# show plot
plt.show()

# find monthly utilization rate for each name 

util_month = df.groupby(['name', 'month'])['hrs', 'Bhours', 'contractor'].sum().reset_index()
util_month['contractor'] = util_month['contractor'].apply(lambda x: 1 if x > 0 else x)
util_month['utilization'] = util_month['Bhours'] / util_month['hrs']

corr = util_month['contractor'].corr(util_month['utilization'])

# find daily utilization rate for each name 

util_day = df.groupby(['name', 'date'])['hrs', 'Bhours', 'contractor'].sum().reset_index()
util_day['contractor'] = util_day['contractor'].apply(lambda x: 1 if x > 0 else x)
util_day['utilization'] = util_day['Bhours'] / util_day['hrs']

corr2 = util_day['contractor'].corr(util_day['utilization'])

## PLOT: Top 10 Projects by Hours and Utilization Rate
    
# group the data by project, find utilization rate for each name within the timespan of the dataset  
util_proj = df.groupby('project')['hrs', 'Bhours', 'contractor'].sum()
util_proj['contractor'] = util_proj['contractor'].apply(lambda x: 1 if x > 0 else x)
util_proj['utilization'] = util_proj['Bhours'] / util_proj['hrs']

# reset the index to make name a regular column
util_proj.index.project = 'project'
util_proj = util_proj.reset_index()

util_proj = util_proj.sort_values(by = 'hrs', ascending = False)
top_10 = util_proj.head(10)

# Create a figure with two y-axes
fig, ax1 = plt.subplots()

# Create a bar chart for the hours column
ax1.bar(top_10['project'], top_10['hrs'], color='cornflowerblue')
plt.xticks(rotation=90)
ax1.set_xlabel('Project Name')
ax1.set_ylabel('Hours')
ax1.tick_params(axis='y', labelcolor='k')

# Create a second y-axis for the utilization rate column
ax2 = ax1.twinx()
ax2.plot(top_10['project'], top_10['utilization'], color='r', marker='o')
ax2.set_ylabel('Utilization Rate')
ax2.tick_params(axis='y', labelcolor='r')

# Add a title to the plot
ax1.grid(False)
ax2.grid(False)
plt.title('Top 10 Projects Ranked by Hours Spent')

# Filter the DataFrame to only include the 'internal' project
util_proj1 = df.groupby('project')['hrs', 'Bhours', 'contractor', 'task'].sum()
internal_df = df.loc[df['project'] == 'Internal']

# Create a bar graph using the filtered DataFrame
task_hours = internal_df.groupby('task')['hrs'].sum()
task_hours = task_hours.sort_values(ascending = False)

# Create a bar graph using the task_hours Series
task_hours.plot(kind='bar', color = 'r')
plt.xlabel('Task')
plt.ylabel('Hours')
plt.grid(False)
plt.title('Hours Spent On Tasks For Internal Project')
plt.show()

# Create plot for top 10 individuals who have the highest utilization rate

util = util.sort_values(by = 'utilization', ascending = False)
top10_indi = util.head(10)

# Create a figure with two y-axes
fig2, ax2 = plt.subplots()

# Create a bar chart for the hours column
ax2.bar(top10_indi['name'], top10_indi['hrs'], color='darkslategrey')
plt.xticks(rotation=90)
ax2.set_xlabel('Name')
ax2.set_ylabel('Hours')
ax2.tick_params(axis='y', labelcolor='k')

# Create a second y-axis for the utilization rate column
ax3 = ax2.twinx()
ax3.plot(top10_indi['name'], top10_indi['utilization'], color='r', marker='o')
ax3.set_ylabel('Utilization Rate')
ax3.tick_params(axis='y', labelcolor='r')
ax3.set_ylim(0, 1.1)
ax2.grid(False)
ax3.grid(False)
plt.title('Top 10 Individuals Ranked by Utilization Rate')

# Create plot for top 10 individuals who have the lowest utilization rate

util = util.sort_values(by = 'utilization', ascending = True)
top10_indi = util.head(10)

# Create a figure with two y-axes
fig2, ax2 = plt.subplots()

# Create a bar chart for the hours column
ax2.bar(top10_indi['name'], top10_indi['hrs'], color='cadetblue')
plt.xticks(rotation=90)
ax2.set_xlabel('Name')
ax2.set_ylabel('Hours')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_ylim(-20, 1300)

# Create a second y-axis for the utilization rate column
ax3 = ax2.twinx()
ax3.plot(top10_indi['name'], top10_indi['utilization'], color='r', marker='o')
ax3.set_ylabel('Utilization Rate')
ax3.tick_params(axis='y', labelcolor='r')
ax3.set_ylim(-0.05, 1.1)
ax2.grid(False)
ax3.grid(False)
plt.title('Bottom 10 Individuals Ranked by Utilization Rate')