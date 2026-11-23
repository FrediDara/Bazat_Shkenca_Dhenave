import pandas as pd
import numpy as np
from itertools import combinations

df_raw = pd.read_csv("groceries_dataset.csv", header=None)
trans = df_raw.values.tolist()    # Kthimi ne List Transaksioni
trans = [ [item for item in trans if isinstance(item, str)]
         for trans in trans ]   # Largimi i vlerave NaN 
print(f"\nTotali i transaksioneve : {len(trans)}")

# --- Krijimi i Matrices binare te Transaksionit ---
items = sorted({item for t in trans for item in t})

df_bin = pd.DataFrame([ {item: (item in t) for item in items}
                         for t in trans]).astype(int)

print("\nMatrica binare (rastet):")
print(df_bin.iloc[:5, :10])

def correlation_coefficient(df, item1, item2):
    A, B = df[item1], df[item2]
    p_A, p_B, p_AB = A.mean(), B.mean(), (A & B).mean()
    numerues = p_AB - (p_A * p_B)
    emerues = np.sqrt(p_A * p_B * (1 - p_A) * (1 - p_B))
    
    if emerues == 0:
        return 0
    return numerues / emerues

# pragu i korrelacionit te forte
pragu = 0.3  
cifte_forta = []

for item1, item2 in combinations(items, 2):
    phi = correlation_coefficient(df_bin, item1, item2)
    if abs(phi) >= pragu:
        cifte_forta.append((item1, item2, round(phi, 3)))

print(f"\nCiftet e korreluara fort (|φ| ≥ {pragu}):")
for i1, i2, val in cifte_forta:
    print(f"{i1} ↔ {i2}  φ = {val}")
print(f"\nU gjeten: {len(cifte_forta)} cifte te forta")
