# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:31:15 2026

@author: Fredi
"""

from pycaret.datasets import get_data
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Load data
data = get_data('france')

# matrica e transaksioneve 
basket = data.groupby(['InvoiceNo', 'Description']).size().unstack(fill_value=0)
basket = (basket > 0).astype(int)

# Grup-artikujt e shpeshte
items = apriori(basket, min_support=0.065, use_colnames=True)

# Rregullat e shoqerimit
arules = association_rules(items, metric='confidence', min_threshold=0.50)

print(arules[['antecedents','consequents','support','confidence','lift']].head())

    
# ============================================================
# 1. Zgjedhja e 10 rregullave me te mira sipas Besimit
# ============================================================

top10 = arules.nlargest(15, 'confidence').copy()

# ============================================================
# 2. Funksion per krijimin e etiketave te shkurtra
# ============================================================

def short_name(item, words=3):
    """Mban vetem disa fjale te para te emrit te nje produkti."""
    return ' '.join(str(item).split()[:words]).title()


# ============================================================
# 3. Krijimi i grafikut 2D 
# ============================================================

plt.figure(figsize=(6, 4))

sc = plt.scatter(top10['support'], top10['confidence'], c=top10['lift'], # 'support'
    cmap='RdYlGn', # All points have the same size
    s=80, alpha=0.85, edgecolors='black',linewidths=0.6)


# ============================================================
# 4. Add short product-name labels
# ============================================================

for _, row in top10.iterrows():
    # Items from both sides of the rule
    items = list(row['antecedents']) + list(row['consequents'])
    # Remove duplicated items
    items = list(dict.fromkeys(items))
    # Shorten each product name
    label = short_name(items[0], 1) #''.join(short_name(items[0], 1)),# for item in items)
    plt.annotate(label,(row['support'], row['confidence']),
        xytext=(6, 6),textcoords='offset points',fontsize=9)

# ============================================================
# 5. Format the graph
# ============================================================

plt.xlabel('Mbeshtetja', fontsize=14)
plt.ylabel('Besimi', fontsize=14)
plt.title('Rregullat e shoqerimit',fontsize=18,fontweight='bold')
# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('Lift', fontsize=13)
# Light grid
plt.grid(alpha=0.15)
plt.tight_layout()
plt.show()    



# GRAFIKU 3D: Support vs Confidence vs Lift
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')
scatter_3d = ax.scatter(
    arules['support'], 
    arules['confidence'], 
    arules['lift'], 
    c=arules['lift'], 
    cmap='plasma', 
    s=50, 
    depthshade=True
)
fig.colorbar(scatter_3d, label='Lift')
ax.set_xlabel('Mbeshtetja')
ax.set_ylabel('Besimi')
ax.set_zlabel('Ngritja')
ax.set_title('Grafiku 3D - Metrikat e shoqerimit')
plt.show()