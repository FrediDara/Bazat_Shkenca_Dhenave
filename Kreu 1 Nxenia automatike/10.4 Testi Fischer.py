

# Importimi i bibliotekave
from scipy.stats import  f_oneway

# Krijimi i te dhenave
# Performance ku vendoset secili nga vajrat e motorit
performance1 = [89, 89, 88, 78, 79]
performance2 = [93, 92, 94, 89, 88]
performance3 = [89, 88, 89, 93, 90]
performance4 = [81, 78, 81, 92, 82]

# Kryerja e ANOVA-s se njeanshme
f_oneway(performance1, performance2, performance3, performance4)

