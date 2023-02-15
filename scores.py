import argparse
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
parser = argparse.ArgumentParser(description='EDOS')
parser.add_argument('--task', type=str, default= '1', help='task number')     
parser.add_argument('--submission', type=str, default='1', help='psubmission number')        

 

args = parser.parse_args()
train = pd.read_csv('submissions/'+args.task+'-'+args.submission'.tsv', delimiter="\t")
labels=df_pred['label']
   
y_true=df_pred['xx']
    
target_names=['neutral','positive','negative']    
print(classification_report(y_true, labels, target_names=target_names))
    
    
from datetime import datetime
f = open("/home/sanala/Juputer try/afrisenti/results-taskA/"+args.task+'-'+args.submission, 'a')
f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
f.write("```\n")
f.write(classification_report(y_true, labels, target_names=target_names))
f.write("```\n")
tn, fp, fn, tp =confusion_matrix(y_true, labels).ravel()
f.write(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}')
f.write("```\n")
f.close()
  
