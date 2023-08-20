# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 20:39:59 2023

@author: Admin
"""
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Admin/Desktop/thesis/امار/HD/Ranked abs of Rel Diffs-2 percents.csv")
print(df.head())

#dff=df[['G 3-Z 1', 'G 5-Z 1', 'G 8- Z 1']]
#dff.head()

longdff=pd.melt(df, var_name='Criteria', value_name= 'value')
longdff.head()

import scikit_posthocs as sp
sp=sp.posthoc_dunn(longdff, val_col='value', group_col='Criteria', p_adjust='fdr_tsbky')


import seaborn as sns

#cor=df.corr()
#cor
mask=np.triu(np.ones_like(sp))
sns.heatmap(data=sp, annot=True, mask=mask , fmt='.1g', annot_kws={'fontsize': 8 , 'fontfamily': 'serif'},linewidth=1, xticklabels=False)

#df1=df.pivot('recon set',)