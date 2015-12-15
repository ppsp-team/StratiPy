# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 00:09:51 2015

@author: aman
"""

import glob,pandas,pickle

ctype = 'OV'
indir = '../data/' + ctype
inmaf = glob.glob(indir+'/Somatic_Mutations/*/*/*.maf')
data = {}
for maf in inmaf:
    this = pandas.DataFrame.from_csv(maf,sep='\t',index_col=None)
    cols = {'Entrez_Gene_Id':'gene','Variant_Classification':'type','Tumor_Sample_Barcode':'sample'}
    this = this.rename(columns = cols)
    this = this[cols.values()]
    this['sample'] = this['sample'].apply(lambda x:'-'.join(x.split('-')[:3]))
    skip = ['Silent']
    this = this[~this['type'].isin(skip)][['sample','gene']]
    this['flag'] = 1
    q = this.groupby(['sample','gene'])['flag'].sum().reset_index()
    q = q.pivot('sample','gene','flag').fillna(0)
    name = maf.split('/')[-3]
    data[name] = q

followup = glob.glob(indir+'/Clinical/Biotab/nationwidechildrens.org_clinical_follow_up*.txt')
allthis = []
for i in followup:
    if i.find('nte')>0:
        continue
    this = pandas.DataFrame.from_csv(i,sep='\t',index_col=None)[['bcr_patient_barcode','vital_status','last_contact_days_to','death_days_to']].ix[2:]
    allthis.append(this)
allthis = pandas.concat(allthis,axis=0)
allthis = allthis[allthis['vital_status'].isin(['Alive','Dead'])]
allthis['event'] = allthis['vital_status'].apply(lambda x:int(x=='Dead'))
allthis['time'] = allthis.apply(lambda x:x['death_days_to'] if x['event'] else x['last_contact_days_to'],axis=1)
allthis = allthis[~(allthis['time']=='[Not Available]')]
allthis = allthis[['bcr_patient_barcode','event','time']]
allthis.columns = ['sample','event','time']
allthis = allthis[allthis['time']>0]
allthis = allthis.groupby('sample').apply(lambda x:x.groupby('event').max().reset_index())
allthis = allthis.groupby('sample').apply(lambda x:x[x['event']==1] if x['event'].max()==1 else x)
allthis = allthis[['sample','event','time']]
data['phenotype'] = allthis

f = open('../res/'+ctype+'.data','w')
pickle.dump(data,f)
f.close()