import argparse
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
parser = argparse.ArgumentParser(description='EDOS')
parser.add_argument('--task', type=str, default= 'pcm', help='task number')     
parser.add_argument('--submission', type=str, default='3', help='psubmission number')        

 

args = parser.parse_args()
train = pd.read_csv('res/'+args.task+'_'+args.submission+'.tsv', delimiter="\t")
test = pd.read_csv('res/'+args.task+'_test_gold_label.tsv', delimiter="\t")


labels=train['label']
   
y_true=test['label']
    
target_names=['neutral','positive','negative']    
print(classification_report(y_true, labels, target_names=target_names))
    
    
from datetime import datetime
f = open("/home/sanala/Juputer try/afrisenti/TaskC/results-taskC/"+args.task+'_'+args.submission, 'a')
f.write(f"\n {datetime.today().strftime('%Y-%m-%d %H:%M:%S')} \n")
f.write("```\n")
f.write(classification_report(y_true, labels, target_names=target_names,digits=4))
f.write("```\n")
#f.write(confusion_matrix(y_true, labels).ravel())
f.write("```\n")
f.close()
  
