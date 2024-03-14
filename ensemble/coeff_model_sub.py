import pandas as pd

sub1 = pd.read_csv('sub_ens.csv')
sub2 = pd.read_csv('sub_mult.csv')
sub3 = pd.read_csv('sub_casc.csv')

for i in range(len(sub1)):
    for x in sub1.columns[1:]:
        if x != 'id':
            sub2.loc[i, x] = sub1[x][i]*(1/6) + sub2[x][i]*(1/2) + sub3[x][i]*(1/3)
sub2.to_csv('misishunters_submition.csv', index=False)