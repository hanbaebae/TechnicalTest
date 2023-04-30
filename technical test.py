# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('customer_transactions.csv')

# Data preprocessing and cleaning
# Drop any duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Drop unnecessary columns
df.drop(['customer_id', 'transaction_id', 'date'], axis=1, inplace=True)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['gender', 'product_category'])

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(df_scaled)

# Add cluster labels to original dataframe
df['cluster'] = kmeans.labels_

# Visualization
plt.scatter(df['age'], df['total_spend'], c=df['cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Total Spend')
plt.show()

# Analyze the characteristics of each segment and provide insights and recommendations for the retail store to improve their marketing and sales strategies.
