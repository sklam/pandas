from __future__ import print_function

import pandas as pd
import numpy as np
import hsagrouper

nelem = 10
numgroup = 4
df = pd.DataFrame({'a': np.random.randint(0, numgroup, nelem).astype(np.int32),
                   'b': np.arange(nelem, dtype=np.float32)})
print(df)

print('default'.center(30, '-'))
for i, group in df.groupby('a'):
    print('+++', i)
    print(group)

print('custom'.center(30, '-'))

groupby = df.groupby(hsagrouper.HSAGrouper('a'))
print(groupby.grouper)
for i, group in groupby:
    print('+++', i)
    print(group)
