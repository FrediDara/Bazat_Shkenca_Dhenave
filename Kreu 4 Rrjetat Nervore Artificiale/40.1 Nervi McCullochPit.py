import numpy as np
np.random.seed(seed=0)
I = np.random.choice([0,1], 3)
# krijimi i vektorit te rastit I, me vlera nga {0,1}
W = np.random.choice([0,1], 3)
# krijimi i vektorit te rastit W, me vlera nga {0,1}
print(f'Vektori hyres:{I}, \nVektori peshe:{W}')

# Hapi 2: Llogaritja e shumës së peshuar (prodhimit numerik) të vektorit hyrës dhe peshave.
dot = I @ W
print(f'Shuma e peshuar: {dot}')

# Hapi 3: Përcaktimi i funksionit të aktivizimit prag binar
def pragu_binar(dot: int, T: float) -> int:
    '''Kthen rezultatin e pragut binar'''
    if dot >= T:
        return 1 
    else:
        return 0

# Hapi 4: Llogaritja e rezultatit bazuar në vlerën e pragut
T = 1
aktivizimi = pragu_binar(dot, T)
print(f'\nAktivizimi per pragun T = {T}: {aktivizimi }')

#vendosur në T = 1. Meqë Shuma = 2, nervi aktivizohet. Nëse e rrisim pragun në T = 3, nervi nuk do të ndizet.

T = 3
aktivizimi = pragu_binar(dot, T)
print(f'Aktivizimi pragun T = {T}: {aktivizimi }')
