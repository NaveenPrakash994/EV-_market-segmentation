#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")




# In[2]:


#Load dataset
ev_sales_data=pd.read_csv('Ev Sales.csv')


# In[10]:


ev_sales_data.head()


# In[3]:


ev_sales_data.info()


# In[4]:


ev_sales_data.describe()


# In[5]:


# Drop rows with missing values
ev_sales_data_cleaned = ev_sales_data.dropna()


# In[6]:


# Splitting the 'YEAR' column into 'Month' and 'Year'
ev_sales_data_cleaned['Month'], ev_sales_data_cleaned['Year'] = ev_sales_data_cleaned['YEAR'].str.split('-', 1).str
ev_sales_data_cleaned['Year'] = '20' + ev_sales_data_cleaned['Year']

# Convert 'Year' to integer
ev_sales_data_cleaned['Year'] = ev_sales_data_cleaned['Year'].astype(int)

# Display the cleaned data with new columns
ev_sales_data_cleaned.head()


# In[7]:


# Calculating correlations between different vehicle types
correlation_matrix = ev_sales_data_cleaned[['2 W', '3 W', '4 W', 'BUS']].corr()

# Creating a heatmap to visualize the correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Between Vehicle Sales')
plt.show()


# In[8]:


import matplotlib.pyplot as plt

# Group data by year and sum up the totals
yearly_sales = ev_sales_data_cleaned.groupby('Year').sum()

# Plotting the total EV sales by year
plt.figure(figsize=(10, 6))
plt.plot(yearly_sales.index, yearly_sales['TOTAL'], marker='o')
plt.title('Total EV Sales per Year in India')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

yearly_sales['TOTAL']


# In[9]:


# Group data by month to see average monthly trends across years
monthly_sales = ev_sales_data_cleaned.groupby('Month').mean()

# Create a sorted list of months for proper sequential plotting
months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_sales = monthly_sales.reindex(months_order)

# Plotting the average monthly EV sales
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['TOTAL'], marker='o')
plt.title('Average Monthly EV Sales in India')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.grid(True)
plt.show()

monthly_sales['TOTAL']


# In[10]:


# Extracting yearly sales data for each vehicle type
yearly_category_sales = ev_sales_data_cleaned.groupby('Year').sum()[['2 W', '3 W', '4 W', 'BUS']]

# Plotting the sales trends for each vehicle category
plt.figure(figsize=(12, 8))
for category in yearly_category_sales.columns:
    plt.plot(yearly_category_sales.index, yearly_category_sales[category], marker='o', label=category)

plt.title('Yearly Sales by Vehicle Category in India')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

yearly_category_sales


# In[11]:


# Plotting histograms for each vehicle category to analyze the distribution of sales
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution of Sales Across Vehicle Categories')

# Histogram for 2-wheeled vehicles
axes[0, 0].hist(ev_sales_data_cleaned['2 W'], bins=15, color='blue', alpha=0.7)
axes[0, 0].set_title('2-Wheeled Vehicles')
axes[0, 0].set_xlabel('Sales')
axes[0, 0].set_ylabel('Frequency')

# Histogram for 3-wheeled vehicles
axes[0, 1].hist(ev_sales_data_cleaned['3 W'], bins=15, color='green', alpha=0.7)
axes[0, 1].set_title('3-Wheeled Vehicles')
axes[0, 1].set_xlabel('Sales')
axes[0, 1].set_ylabel('Frequency')

# Histogram for 4-wheeled vehicles
axes[1, 0].hist(ev_sales_data_cleaned['4 W'], bins=15, color='red', alpha=0.7)
axes[1, 0].set_title('4-Wheeled Vehicles')
axes[1, 0].set_xlabel('Sales')
axes[1, 0].set_ylabel('Frequency')

# Histogram for buses
axes[1, 1].hist(ev_sales_data_cleaned['BUS'], bins=15, color='purple', alpha=0.7)
axes[1, 1].set_title('Buses')
axes[1, 1].set_xlabel('Sales')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[20]:


# Plotting sales trends over time for each vehicle category
plt.figure(figsize=(14, 8))
for category in ['2 W', '3 W', '4 W', 'BUS']:
    plt.plot(ev_sales_data_cleaned['Year'], ev_sales_data_cleaned[category], marker='o', label=category)

plt.title('Sales Trends Over Time by Vehicle Category')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Selecting features for clustering
features = ev_sales_data_cleaned[['2 W', '3 W', '4 W', 'BUS']]

# Normalizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Finding the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the WCSS to find the elbow point
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()


# In[13]:


# Performing K-means clustering with K=3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Adding the cluster labels to the original data
ev_sales_data_cleaned['Cluster'] = clusters

# Analyzing the clusters by calculating mean values for each cluster
cluster_analysis = ev_sales_data_cleaned.groupby('Cluster').mean()
cluster_analysis


# In[ ]:




