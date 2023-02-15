
import pandas as pd
import os
DataPath= '/home/sanala/Juputer try/afrisenti/TaskA/train'   #change it to your path
#data
for i in ['am', 'dz' ,'ha', 'ig', 'kr' ,'ma' ,'pcm' ,'pt' ,'sw' ,'ts' ,'twi' ,'yo'] :
    train_data = pd.read_csv(os.path.join(DataPath,i+'_train.tsv'), sep='\t')
    train, valid = train_test_split(train_data, test_size=0.2, random_state=0, stratify=train_data[['label']])
    train.to_csv(os.path.join(DataPath,i+'_train_new.tsv', sep='\t')
    valid.to_csv(DataPath,i+'_dev_new.tsv', sep='\t')

