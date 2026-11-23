

# Importimi i bibliotekave
import numpy as np
import scipy.stats as stats

# Krijimi i te dhenave
grupi1 = np.array([14, 15, 15, 16, 13, 8, 14, 17, 16, 14, 19, 
                   20, 21, 15, 15, 16, 16, 13, 14, 12])
grupi2 = np.array([15, 17, 14, 17, 14, 8, 12, 19, 19, 14, 17, 
                   22, 24, 16, 13, 16, 13, 18, 15, 13])


# Printimi i variances se dy grupeve
print(np.var(grupi1), np.var(grupi2))

# Kryerja e testit te Studentit me varianca te njejta
stats.ttest_ind(a=grupi1, b=grupi2, equal_var=True)
