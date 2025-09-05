# `CRISP-ML(Q)` process model describes six phases:
# 
# 1. Business and Data Understanding
# 2. Data Preparation
# 3. Model Building
# 4. Model Evaluation
# 5. Deployment
# 6. Monitoring and Maintenance


# Objective: Identify high-impact book purchase patterns using Association Rule Mining to boost Kitabi Duniya’s sales by 25%.

# Constraint: Insights must be derived from limited transaction data and implemented within budget, brand, and store layout limitations.


# Success Criteria

# Business Success Criteria : 25% increase in footfall and revenue, improved customer retention
# Machine Learning Criteria : NA
# Economic Success Criteria : ROI from campaigns > cost of implementation, increased average basket size


# Code Modularity

# pip install mlxtend

# Import NumPy for numerical computations
import numpy as np

# Import pandas for data manipulation and analysis
import pandas as pd

# Import matplotlib for visualizing data and patterns
import matplotlib.pyplot as plt

# Import Apriori algorithm and association rule generator from mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# Import transaction encoder to convert list-based data into a format suitable for mining
from mlxtend.preprocessing import TransactionEncoder

# Import SQLAlchemy to establish connection between Python and MySQL
from sqlalchemy import create_engine

# Load book transaction data from CSV file into a pandas DataFrame
data = pd.read_csv(r"C:\Users\bomma\Downloads\Assignments Questions\Data Science\book.csv")

# Define MySQL username
user = 'root'

# Define MySQL password
pw = 'venkat1236'

# Define target database name
db = 'ML_assignments'

# Create a connection engine to MySQL using pymysql driver
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

# Upload the DataFrame to MySQL as a table named 'book_store'; replace if it already exists
data.to_sql('book_store', con=engine, if_exists='replace', chunksize=1000, index=False)

# Define SQL query to retrieve all records from the 'book_store' table
sql = 'select * from book_store;'

# Execute SQL query and load the result into a new DataFrame
book_store = pd.read_sql_query(sql, con=engine)

book_store.columns

book_store.shape

book_store.info()



# Convert binary matrix to list of item names per transaction
books_1 = []
for index, row in book_store.iterrows():
    transaction = [col for col in book_store.columns if row[col] == 1]
    books_1.append(transaction)

# Encode transactions into boolean matrix
TE = TransactionEncoder()

# Apply TransactionEncoder to convert book transactions into a one-hot encoded NumPy array
encoded_5 = TE.fit_transform(books_1)


transformed_data = transformed_data.columns.str.capitalize()

# Convert encoded array into a DataFrame with readable column names from the encoder
transformed_data = pd.DataFrame(encoded_5, columns=TE.columns_)

# Check the dimensions of the transformed dataset (rows = transactions, columns = items)
transformed_data.shape

# Display the item names (book titles) used as columns in the encoded DataFrame
transformed_data.columns

# Label the index to indicate each row represents a book transaction
transformed_data.index.name = 'Book'

# View the final one-hot encoded DataFrame for further analysis
transformed_data


# Generate frequent itemsets using the Apriori algorithm with a minimum support threshold
appi = apriori(transformed_data, min_support=0.0075, max_len=6, use_colnames=True)

# Display the frequent itemsets generated
appi

# Derive association rules from frequent itemsets using lift as the evaluation metric
lift_ratio = association_rules(appi, metric='lift', min_threshold=1)

# Define a helper function to sort items in a rule for consistent duplicate detection
def clean_data(i):
    return sorted(list(i))

# Apply sorting to both antecedents and consequents, then combine them into a single list
new_list = lift_ratio.antecedents.apply(clean_data) + lift_ratio.consequents.apply(clean_data)

# Sort the combined item list to ensure uniformity across rules
new_list = new_list.apply(sorted)

# Convert the Series of item lists into a standard Python list
new = list(new_list)

# Remove duplicate rules by converting each list to a tuple, applying set(), and converting back to list
unique = [list(m) for m in set(tuple(i) for i in new)]

# Initialize an empty list to store indices of unique rules
index = []

# Populate the index list with positions of each unique rule
for i in unique:
    index.append(unique.index(i))

# Filter the original rules to retain only those with unique item combinations
no_ren = lift_ratio.iloc[index, :]

# Display the deduplicated set of association rules
no_ren

# Sort the deduplicated rules by lift in descending order to prioritize strongest associations
top = no_ren.sort_values('lift', ascending=False)

# Display the top association rules based on lift
top



# Convert antecedents from frozenset to string format for easier text manipulation
top['antecedents'] = top['antecedents'].astype('string')

# Convert consequents from frozenset to string format for easier text manipulation
top['consequents'] = top['consequents'].astype('string')

# Remove the 'frozenset({' prefix from antecedents to clean up the display
top['antecedents'] = top['antecedents'].str.removeprefix('frozenset({')

# Remove the closing '})' suffix from antecedents for readability
top['antecedents'] = top['antecedents'].str.removesuffix('})')

# Remove the 'frozenset({' prefix from consequents to clean up the display
top['consequents'] = top['consequents'].str.removeprefix('frozenset({')

# Remove the closing '})' suffix from consequents for readability
top['consequents'] = top['consequents'].str.removesuffix('})')

# Display the cleaned top association rules with readable antecedents and consequents
top


# Check for missing (null) values in each column of the DataFrame
top.isnull().sum()

# Import NumPy for numerical operations and handling special values like inf
import numpy as np

# Check for any infinite values (inf or -inf) in numeric columns
np.isinf(top.select_dtypes(include=[float, int])).sum()

# Replace all inf and -inf values with NaN to make the DataFrame SQL-compatible
top.replace([np.inf, -np.inf], np.nan, inplace=True)

# Recheck to confirm all infinite values have been removed
np.isinf(top.select_dtypes(include=[float, int])).sum()

# Export the cleaned DataFrame to a SQL table named 'books_store_association_rule'

top.to_sql('books_store_association_rule', con=engine, if_exists='replace', chunksize=1000, index=False)

