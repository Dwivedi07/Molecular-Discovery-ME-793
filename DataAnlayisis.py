import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns


# Read the data from the text file and split by space
with open('data_test.txt', 'r') as file:
    lines = file.readlines()
    data_list = [line.strip().split() for line in lines]

# Create DataFrame from the list
data_df = pd.DataFrame(data_list, columns=['SMILES', 'Value'])

# Display the DataFrame
print(data_df.head())

# Save the DataFrame to a new CSV file
data_df.to_csv('data_processed.csv', index=False)


# Add a new column for molecular weight
data_df['Molecular_Weight'] = data_df['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))

# Display the first few rows of the updated data
print(data_df.head())

# Summary statistics for molecular weight
print(data_df['Molecular_Weight'].describe())

# Histogram of molecular weight
plt.figure(figsize=(10, 6))
sns.histplot(data_df['Molecular_Weight'], kde=True)
plt.title('Histogram of Molecular Weight')
plt.xlabel('Molecular Weight')
plt.ylabel('Frequency')
plt.show()

# Descriptive statistics
print("Descriptive Statistics of Photovoltaic Efficiency:")
print(data_df['Value'].describe())
print()

# Histogram of photovoltaic efficiency
plt.figure(figsize=(10, 6))
sns.histplot(data_df['Value'], kde=True, color='skyblue')
plt.title('Histogram of Photovoltaic Efficiency')
plt.xlabel('Photovoltaic Efficiency')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Boxplot of photovoltaic efficiency
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_df['Value'], color='lightgreen')
plt.title('Boxplot of Photovoltaic Efficiency')
plt.xlabel('Photovoltaic Efficiency')
plt.grid(True)
plt.show()

# Pairwise Scatter Plot (if additional features are available)
# For example, if there are multiple features, you can visualize the relationship between photovoltaic efficiency and another feature.
# sns.pairplot(data_df, vars=['Feature1', 'Feature2', 'Value'], diag_kind='kde')
# plt.show()

# Fit a probability distribution to the photovoltaic efficiency data
plt.figure(figsize=(10, 6))
sns.histplot(data_df['Value'], kde=True, color='skyblue', stat='density', label='Histogram')

# Fit a normal distribution to the data
mu, std = norm.fit(data_df['Value'])
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label=f'Fit: $\mu$={mu:.2f}, $\sigma$={std:.2f}')

plt.title('Probability Distribution of Photovoltaic Efficiency')
plt.xlabel('Photovoltaic Efficiency')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()