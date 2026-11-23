from collections import defaultdict
from itertools import combinations

def ais_alg(trans, min_support):
   # Konvertimi i transaksioneve ne lista te renditura
   trans = [sorted(t) for t in trans]

   # Numri i grup-artikujve 1
   item_count = defaultdict(int)
   for t in trans:
      for item in t:
         item_count[(item,)] += 1

   # Grup-artikujt e shpeshte 1
   L = {itemset: count for itemset, count in item_count.items() if count >= min_support}

   all_frequents = dict(L)
   k = 2

   while L:
      cand_count = defaultdict(int)
      for t in trans:
         t_set = set(t)
         # Gjetja e grup-artikuj te shpeshte (k-1) ne transaksion
         for itemset in L:
            if set(itemset).issubset(t_set):
               last_item = itemset[-1]
               for item in t:
                  if item > last_item:
                      cand = tuple(sorted(itemset + (item,)))
                      if set(cand).issubset(t_set):
                        cand_count[cand] += 1
      # filtrim sipas support
      L = {c: count for c, count in cand_count.items() if count >= min_support}
      all_frequents.update(L)
      k += 1

   return all_frequents

# Te dhënat 
trans = [{"bread", "milk"}, {"bread", "diaper", "beer", "egg"},
    {"milk", "diaper", "beer", "cola"},
    {"bread", "milk", "diaper", "beer"},
    {"bread", "milk", "diaper", "cola"}]

min_support = 2

frequent_itemsets = ais_alg(trans, min_support)

print("Grup-artikujt e shpeshte:")
for itemset, count in frequent_itemsets.items():
    print(itemset, "mbeshtetja:", count)
