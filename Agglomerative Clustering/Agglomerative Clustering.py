# Hierarchical Clustering

# CRISP-ML(Q) process model describe six phases:
    
# 1. Business understanding and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance.


# Success Criteria: 
    
    
# Business Success Criteria: Increase the operational efficiency by 10% to 12% by segmenting the Airlines.
# ML Success Criteria: Achieve a Silhouette coefficient of at least 0.7.
# Economic Success Criteria: The airline companies will see an increase in revenues by at least 8% (hypothetical numbers).


# Code Modularity
# pip install sqlalchemy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine, text
from urllib.parse import quote
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OrdinalEncoder # Importing the OrdinalEncoder class from the sklearn.preprocessing module

# Import hierarchical clustering tools
from scipy.cluster.hierarchy import linkage, dendrogram

# Import Agglomerative Clustering model
from sklearn.cluster import AgglomerativeClustering

# Import general evaluation metrics
from sklearn import metrics

# Optional: Import clusteval for advanced cluster validation
from clusteval import clusteval


# Reading the dataset from an Excel file into a Pandas DataFrame.

data = pd.read_csv(r"C:\Users\bomma\Downloads\Assignments Questions\Data Science\Data Set (5)\AirTraffic_Passenger_Statistics.csv")

# Credentials to connect to Database.

user = 'root'
pw = 'venkat1236'
db = 'ML_assignments'

engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

data.to_sql('airtraffic', con = engine, if_exists = 'replace', chunksize = 1000,index = False)

sql = 'select * from airtraffic;'
df = pd.read_sql_query(text(sql),engine.connect())


# Data types

df.info()

# EXPLORATORY DATA ANALYSIS (EDA) / DESCRIPTIVE STATISTICS

df.describe()

# Shape of the Dataset

df.shape

# Count unique non-null values

df.nunique()

# Check how many missing values exist in each column
df.isnull().sum()

# Drop columns that are either redundant or not needed for analysis
df.drop(['Activity Period','Operating Airline','Operating Airline IATA Code','GEO Region','Year','Month'], axis=1, inplace=True)

# Display updated column data types and non-null counts
df.info()

# Show the shape of the dataset after dropping columns (rows, columns)
df.shape

# Count total number of duplicate rows in the dataset
duplicates = df.duplicated().sum()
duplicates

# Auto EDA

# D-Tale

import dtale

# Display the DataFrame using D-Tale

d = dtale.show(df,host = 'localhost', port = 8000)

#  Open the browser to view the interactive D-Tale dashboard

d.open_browser()



# Data Preprocessing

# Checking outlier for all columns

df.boxplot(figsize=(12, 6))
plt.show()

df['Passenger Count'].plot.box()   # Boxplot to detect outliers and spread in passenger counts
plt.show()


# Apply Gaussian-based Winsorization (mean ± 2*std) to cap outliers in Passenger Count

# Note: This method did not effectively treat extreme values in this dataset

winsor_passenger = Winsorizer(capping_method='gaussian', tail='both', fold=2, variables=['Passenger Count'])
df_winsor_passenger_count = winsor_passenger.fit_transform(df[['Passenger Count']])

# Visualize Passenger Count distribution after Gaussian Winsorization

sns.boxplot(df_winsor_passenger_count)
plt.title("Passenger Count - Gaussian Winsorization")
plt.show()

# Apply IQR-based Winsorization (1.5*IQR rule) to cap outliers more effectively

winsor_passenger = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Passenger Count'])
df_winsor_passenger_count = winsor_passenger.fit_transform(df[['Passenger Count']])


# Visualize Passenger Count distribution after IQR Winsorization

sns.boxplot(df_winsor_passenger_count)
plt.title("Passenger Count - IQR Winsorization")
plt.show()


# zero variance and near zero variance

# Display data types of all columns

df.dtypes

# Select only numeric columns from the DataFrame
numeric = df.select_dtypes(include = np.number)

# Identify numeric columns with zero variance (all constant values)
numeric.var() == 0

# No columns have zero variance, so all numeric columns provide variability and can be kept
# So no need of removing columns from the dataset


# Type casting

df.dtypes

df.head(10)


# Data types of all columns are retained as-is since they align with the expected formats for analysis.

# No conversions are necessary at this stage, as object types represent categorical or label data appropriately,
# and numeric columns are already in suitable formats for aggregation and modeling.



# Discretization/Binning

# No data type conversions are applied, as all columns are already in suitable formats for analysis.

# Discretization or binning is also not required, since the dataset does not contain continuous variables

# that need to be transformed into categorical bins for modeling or segmentation purposes.

# This preserves the original granularity of the data, which is important for accurate operational insights.


# Dummy Variable Creation

# Here we are using Ordinal Encoding because

# Terminal and Boarding Area contain many distinct categories, but they do not have a meaningful ordinal relationship.

# Ordinal encoding is avoided to prevent introducing artificial order into nominal features.

# Alternative encoding methods (e.g., frequency or target encoding) may be considered depending on the modeling context.

# Ordinal Encoding can handle multiple dimensions or features simultaneously.


from sklearn.preprocessing import OrdinalEncoder
# Apply Ordinal Encoding
cols_to_encode = ['Terminal', 'Boarding Area']
oe = OrdinalEncoder()
df[cols_to_encode] = oe.fit_transform(df[cols_to_encode])


# Import scaler for feature normalization
from sklearn.preprocessing import StandardScaler  

# Initialize the scaler
scaler = StandardScaler()                         

# Scale 'Passenger Count' for uniformity
df['Passenger Count'] = scaler.fit_transform(df[['Passenger Count']])  

# Extract relevant features for clustering

df_cleaned = df[['Terminal', 'Boarding Area', 'Passenger Count']]  

# Preview the cleaned dataset

df_cleaned  


# End of Data Preprocessing

# Save Preprocessed data into SQL Mandatory

# Credentials to connect to Database.

# Define database username

user = 'root'

# Define database password

pw = 'venkat1236'

# Specify target database name
db = 'ML_assignments'

# Create SQLAlchemy engine to connect to MySQL
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Save preprocessed DataFrame to SQL table 'airtraffic_scaled'
df.to_sql('airtraffic_scaled', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# Model Building

# CLUSTERING MODEL BUILDING

# Hierarchical Clustering - Agglomerative Clustering


# Import hierarchical clustering tools
from scipy.cluster.hierarchy import linkage, dendrogram

# Import Agglomerative Clustering model
from sklearn.cluster import AgglomerativeClustering

# Import general evaluation metrics
from sklearn import metrics

# Optional: Import clusteval for advanced cluster validation
from clusteval import clusteval



# Generate dendrogram using Ward linkage method
tree_plot = dendrogram(linkage(df_cleaned, method = 'ward'))

# Set title for dendrogram plot
plt.title('Hierarchical Clustering')

# Label x-axis of dendrogram
plt.xlabel('Cluster Label')

# Label y-axis of dendrogram
plt.ylabel('Euclidean Distance')

# Display dendrogram
plt.show()

# Check data types of DataFrame columns
df.dtypes


# Applying AgglomerativeClustering and grouping data into 3 clusters 

# Initialize Agglomerative Clustering with 3 clusters
model = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')

# Fit model and assign cluster labels to cleaned DataFrame
df_cleaned['Cluster'] = model.fit_predict(df_cleaned)

# View clustered DataFrame
df_cleaned

# Access cluster labels directly from model
model.labels_

# Preview first few rows of clustered data
df_cleaned.head()


# Check the number of data points in each cluster
df_cleaned['Cluster'].value_counts()

# Clusters Evaluation

# Silhouette coefficient:
    
# Import silhouette score function
from sklearn.metrics import silhouette_score

# Calculate silhouette score to evaluate cluster separation
score = silhouette_score(df_cleaned.drop('Cluster', axis = 1), df_cleaned['Cluster'])

# Print silhouette score
print('Silhouette Score :', score)

#Calinski-Harabasz

# Import Calinski-Harabasz score function
from sklearn.metrics import calinski_harabasz_score

# Calculate CH score to assess cluster dispersion
chs = calinski_harabasz_score(df_cleaned.drop('Cluster', axis = 1), df_cleaned['Cluster'])

# Print Calinski-Harabasz score
print('Calinski Harabasz Score : ', chs)


# Davies-Bouldin Index:
    
# Import Davies-Bouldin score function
from sklearn.metrics import davies_bouldin_score

# Calculate DB score to measure intra-cluster similarity
dbs = davies_bouldin_score(df_cleaned.drop('Cluster', axis = 1), df_cleaned['Cluster'])

# Print Davies-Bouldin score
print('Davies Bouldin Score : ', dbs)


# Save final data into Database

# Define database username

user = 'root'

# Define database password

pw = 'venkat1236'

# Specify target database name
db = 'ML_assignments'

# Create SQLAlchemy engine to connect to MySQL
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Save preprocessed DataFrame to SQL table 'airtraffic_scaled'
df_cleaned.to_sql('airtraffic_scaled_full', con=engine, if_exists='replace', index=False)



#Hyperparameter Optimization for Hierarchical Clustering


# Define the grid of hyperparameters to search for optimal clustering

param_grid = {
    'n_clusters': [2, 3, 4],  # Number of clusters to test
    'metric': ['euclidean', 'manhattan', 'cosine'],  # Distance metrics to evaluate
    'linkage': ['ward', 'complete', 'single', 'average']  # Linkage methods for merging clusters
}


# Import GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Import silhouette_score for evaluating clustering quality
from sklearn.metrics import silhouette_score


# Define a custom scoring function for clustering evaluation

def custom_scorer(estimator, X):
    labels = estimator.fit_predict(X)  # Fit model and get cluster labels
    if len(set(labels)) == 1:  # If all points fall into one cluster
        return -1  # Penalize with a low score
    return silhouette_score(X, labels)  # Return silhouette score for valid clustering



# Initialize Agglomerative Clustering
agg_clustering = AgglomerativeClustering()


# Set up GridSearchCV to find the best clustering parameters
grid_search = GridSearchCV(
    estimator=agg_clustering,  # Clustering model to tune
    param_grid=param_grid,     # Hyperparameter combinations to test
    scoring=custom_scorer,     # Use custom silhouette-based scoring
    cv=3                       # Use 3-fold cross-validation (not typical for unsupervised, but accepted here)
)

# Fit GridSearchCV on feature data (excluding cluster labels)

grid_search.fit(df_cleaned.drop('Cluster', axis=1))


# Display the best parameter combination found during grid search
print("Best Parameters:", grid_search.best_params_)


# Display the highest silhouette score achieved
print("Best Silhouette Score:", grid_search.best_score_)


'''
Q1. Why it is important to define the objectives for any Business problem?

Defining objectives ensures clarity, guides the analytical method, aligns stakeholders, 
and sets measurable success criteria — making business problem-solving focused, actionable, and impactful.


Q2. How to maintain the quality of the Machine Learning model developed for the Business problem?

Maintaining ML model quality requires robust validation, drift monitoring, reproducible workflows, and alignment with business KPIs. 
Interpretability and stakeholder feedback ensure the model remains relevant, reliable, and impactful over time.


Q3. What is the first document created/drafted for any ML project?

Project Charter - It defines the problem, objectives, success criteria, and business impact.


Q4. How to load data with multiple sheets?

import pandas as pd

# Load all sheets into a dictionary of DataFrames
excel_file = 'airline_data.xlsx'
all_sheets = pd.read_excel(excel_file, sheet_name=None)

# Access individual sheets by name
routes_df = all_sheets['Routes']
customers_df = all_sheets['Customers']



Q5. What are the Auto EDA techniques?


Pandas Profiling, Sweetviz, D-Tale


Q6. What are four business moments, and what insights we can draw from them?


1. Measures of Central Tendency = Mean,Median,Mode
2. Measures of Dispersion = Variance, Standard Deviation,Range
3. Skewness = Direction of data
4. Kurtosis = Peakedness of Tail


Q7. Write the techniques in data Pre-Processing.

1. Missing Values - handling missing values
2. Duplicates - removing duplicates
3. Outlier Analysis - detecting outliers
4. Zero Varaince - eliminating zero-variance features
5. Type Casting
6. Discretization - Converting Continuous data to Discrete Data
7. Dummy Variable Creation - Converting Categorical data to Numerical Format
8. Transformation - Transforming skewed data
9. Scaling - ensure data quality, consistency, and model readiness for impactful machine learning.



Q8. When we use label encoding and one-hot encoding?


Use label encoding for ordinal variables where order matters,
One-hot encoding for nominal variables to prevent misleading model assumptions.


Q9. What is the technique to remove outliers?

1. Gaussian Limit or Z-Score - Normally Distributed Data
2. IQR limit - Skewed Distributions
3. Mad Median Rule 
4. Percentile or Qunatiles


Q10. What are the techniques to check whether the data is normally distributed or not?

Using visual tools like histograms and Q-Q plots,metrics like skewness and kurtosis.


Q12. How to make data scale-free?


To make data scale-free, apply techniques like standardization, normalization, robust scaling.


Q13. What types of graphs are used to depict the bivariate analysis?

Scatter Plot,Box Plot,Heat Map


Q14. What do you mean by bivariate frequency distribution?

Summarizes the joint frequency of two variables by organizing their paired values into a two-way table.


Q15. Which libraries are used in Hierarchical clustering?

from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import Agglomerative clustering


Q16. What is the difference between Agglomerative clustering and Divisive Clustering?

Agglomerative clustering builds clusters from the bottom up by merging similar data points, 
Divisive clustering starts with one large cluster and splits it recursively. 


Q17. Which metric is used to find distance/similarities between two data points and between a record and a cluster?


Between two Data Points

1.Euclidean Distance
2.Manhattan Distance
3.Minkowski Distance
4.Mahalanobis Distance

Between Record to Cluster and Cluster to CLuster


1.Single Linkage
2.Complete Linkage
3.Average Linkage
4.Ward Linkage


Q18. What are the parameters needed to plot the Dendrogram?

The dendrogram is plotted using a linkage matrix derived from df_pipelined, 
with method="complete" specifying the clustering strategy.


Q19. How to perform cluster evaluation? Which are the techniques used for cluster evaluation?

Metrics like Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.


Q20. What do the Silhouette coefficient, Calinski Harbaz, and Davies-Bouldin Index indicate in hierarchical clustering?


Silhouette coefficient - -1 to +1 
Calinski Harbaz - Higher is Better
Davies-Bouldin - Lower is Better


'''