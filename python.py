# https://www.kaggle.com/datasets/shreyanshverma27/
# online-sales-dataset-popular-marketplace-data?resource=download

import pandas as pd
import numpy as np

df = pd.read_csv('1.csv')
df.tail() # last 5 cols
df.shape  # (rows, cols)
df.columns  # column-names
df.dtypes # data-type of columns (strings-treats as objs)

# accessing data
df['Product Name'] # Access a specific column by name
df.describe() # descriptive statistics 


df[df['Unit Price'] > 998]         # filterations
df.sort_values(by='Unit Price', ascending=True)


# Adding or remove data
df.drop(columns=['Date', 'Units Sold']) # columns
df.drop(index=0) # row


# Add a new column with sample data
# Create an array of alternating 'a' and 'b'
new_column_data = list(range(1, 241))
# print(  len(new_column_data), len(df) )

if len(new_column_data) == len(df):
    df['New Column'] = new_column_data
else:
    print("Length of new column data does not match the length of the DataFrame.")

 


## numpy with data-frame
# Convert 'Units Sold' and 'Total Revenue' columns to NumPy arrays
units_sold_array = df['Units Sold'].values
total_revenue_array = df['Total Revenue'].values
unit_price_array = df['Unit Price'].values

# Perform basic NumPy operations
total_units_sold = np.sum(units_sold_array)
total_revenue = np.sum(total_revenue_array)
average_unit_price = np.mean(unit_price_array)

# Print the results
print("Total Units Sold:", total_units_sold)
print("Total Revenue:", total_revenue)
print("Average Unit Price:", average_unit_price)

import matplotlib.pyplot as plt

x = list(range(1, 241))
y = df['Total Revenue'].values
plt.plot(x, y)
plt.xlabel('Transactions')
plt.ylabel('Revenue Rate')
plt.title('Total Revenue Chart Of Sale Per Transaction')
plt.show()


# regression

 

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

 
 
# Extracting the features (X) and target (y)
X = df[['Transaction ID']]  # Feature
y = df['Total Revenue']  ** 78  # Target

# Creating and fitting the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Getting the coefficients
slope = model.coef_[0]
intercept = model.intercept_

# Printing the coefficients
print("Slope:", slope)
print("Intercept:", intercept)

# Plotting the data points and regression line
plt.scatter(X, y, color='blue', label='Data')  # Data points
plt.plot(X, model.predict(X), color='red', label='Regression Line')  # Regression line

# Adding labels and title
plt.xlabel('Transaction ID')
plt.ylabel('Total Revenue')
plt.title('Linear Regression')

# Adding legend
plt.legend()

# Displaying the plot
plt.show()



