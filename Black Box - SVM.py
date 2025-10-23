# CRISP-ML(Q) process model describes six phases:
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Evaluation
# 5. Model Deployment
# 6. Monitoring and Maintenance


'''
Business Understanding :
    
    Business Problem : A construction firm is planning to invest in developing a suburban locality. 
                       However, the firm faces financial risk if the properties do not sell. 
                       To mitigate this, they seek insights into the area's population density and income levels to assess the viability of their investment.


    Business Objective : Use income and demographic data to assess investment viability.
    
    Business Constraints : Limited data quality, categorical encoding, and model complexity.
    
    
Success Criteria : 
    
    Business : Identify high-income segments and recommend profitable zones
    
    Machine Learning : Achieve ≥80% accuracy with balanced precision and recall.
    
    Economic : Maximize ROI by targeting areas with strong purchasing power.

'''


# Code Madularity

# Load and manipulate structured data
import pandas as pd

# Perform numerical operations and array handling
import numpy as np

# Create visualizations and plots
import matplotlib.pyplot as plt

# Apply transformations to specific columns
from sklearn.compose import ColumnTransformer

# Build modular preprocessing pipelines
from sklearn.pipeline import Pipeline

# Handle outliers using winsorization
from feature_engine.outliers import Winsorizer

# Normalize features to a fixed range
from sklearn.preprocessing import MinMaxScaler

# Connect to MySQL database using SQLAlchemy
from sqlalchemy import create_engine


# Load the CSV file into a DataFrame
data = pd.read_csv(r"C:\Users\bomma\Downloads\Assignments Questions\Data Science\Black Box-SVM\SalaryData_Train.csv")

# Define MySQL credentials
user = 'root'
pw = 'venkat1236'
db = 'ml_assignments'

# Create SQLAlchemy engine to connect to MySQL
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

# Upload DataFrame to MySQL as a new table
data.to_sql('income_data', con=engine, if_exists='replace', chunksize=1000, index=False)

# Define SQL query to retrieve all records
sql = 'SELECT * FROM income_data;'

# Read data back from MySQL into a DataFrame
income = pd.read_sql_query(sql, engine)

# Display column types and non-null counts
income.info()

# Separate features and target
X = income.drop(columns=['Salary'])
y = income['Salary']

# View column names
print(X.columns)

# Drop categorical and target columns for numerical analysis
X = income.drop(columns=[
    'workclass', 'education', 'educationno', 'maritalstatus',
    'occupation', 'relationship', 'race', 'sex', 'native', 'Salary'
])

# Display column info (missing parentheses in your code)
X.info()

# Show descriptive statistics for numerical features
X.describe()

# Check class distribution in target variable
y.value_counts()


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler

# Define the pipeline correctly using tuples
num_pipeline = Pipeline([
    ('winsorize', Winsorizer(capping_method='gaussian', tail='both', fold=1.5)),
    ('scaling', MinMaxScaler())
])


# Apply the pipeline to numerical columns in X
preprocessor = ColumnTransformer([
    ('clean', num_pipeline, X.columns)
])

# Fit and transform the data
clean_data = preprocessor.fit_transform(X)


X1 = pd.DataFrame(clean_data,columns=X.columns)

X1.describe()

X1.to_sql('clean_income_data',con=engine,if_exists='replace',chunksize=1000,index = False)


# Model Building

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Support Vector Classifier for classification tasks
from sklearn.svm import SVC

# Perform randomized hyperparameter search with cross-validation
from sklearn.model_selection import RandomizedSearchCV


# Split the cleaned data into training and testing sets (stratified by target class)
train_X, test_X, train_y, test_y = train_test_split(X1, y, test_size=0.2, stratify=y)

# Initialize and train a linear SVM model
model_linear = SVC(kernel='linear')
model1 = model_linear.fit(train_X, train_y)

# Predict on the test set
pred_test_linear = model_linear.predict(test_X)

# Calculate and display accuracy
accuracy_linear = np.mean(pred_test_linear == test_y)
accuracy_linear


# Initialize base SVM model
model = SVC()

# Define hyperparameter grid for randomized search
parameters = {
    'C': [0.1, 1, 10, 100],               # Regularization strength
    'gamma': [1, 0.1, 0.01, 0.001],       # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel types to explore
}

# Set up randomized search with cross-validation
rand_search = RandomizedSearchCV(
    model, parameters, n_iter=10, n_jobs=3, cv=3,
    scoring='accuracy', random_state=42
)

# Fit randomized search on training data
randomised = rand_search.fit(train_X, train_y)

# View best model and parameters
best_model = randomised.best_estimator_
print("Best Parameters:", randomised.best_params_)

# Predict and evaluate accuracy
pred_test = best_model.predict(test_X)
accuracy = np.mean(pred_test == test_y)
print("Best Model Accuracy:", accuracy)
