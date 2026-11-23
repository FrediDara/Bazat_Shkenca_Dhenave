from pycaret.datasets import get_data
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules

# Load data
data = get_data('france')

# Number of transactions and items
print("Number of transactions:", data['InvoiceNo'].nunique())
print("Number of items:", data['Description'].nunique())

# Create transaction-item matrix
basket = (
    data.groupby(['InvoiceNo', 'Description'])
        .size()
        .unstack(fill_value=0)
)

# Convert quantities to 0/1
basket = (basket > 0).astype(int)

# Find frequent itemsets
frequent_itemsets = apriori(
    basket,
    min_support=0.05,
    use_colnames=True
)

# Generate association rules
arules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.50
)

# Display rules
arules = arules[
    ['antecedents', 'consequents', 'support', 'confidence', 'lift']
]

arules = arules.sort_values(
    'confidence',
    ascending=False
)

print(arules.head(20))


arules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.50
)

plt.figure(figsize=(9, 6))

plt.scatter(
    arules['support'],
    arules['confidence'],
    s=arules['lift'] * 50,
    alpha=0.6
)

plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(9, 7))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    arules['support'],
    arules['confidence'],
    arules['lift'],
    s=40,
    alpha=0.7
)

ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_zlabel('Lift')

ax.set_title('Association Rules - 3D')

plt.show()