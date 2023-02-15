### To run the code for each task use 

>python run_train_ma.py  --model_name  < model name>  --language < language can be one of > --learning_rate < LR > --epoch < no of epochs >  --output_dir < output dir >

-- language can be one of 'am', 'dz' ,'ha', 'ig', 'kr' ,'ma' ,'pcm' ,'pt' ,'sw' ,'ts' ,'twi' ,'yo'

example : 
> python run_train_ma.py  --model_name  'alger-ia/dziribert'  --language 'ma' --learning_rate 5e-5 --epoch 5.0  --output_dir 'alger-ia/dziribert-ma'

### To run the code for prediction on test set for each task use:
>python run_predict_ma.py --model_name < model name > --language < language > --output_dir < output dir >


example:
> python run_predict_ma.py --model_name 'alger-ia/dziribert' --language 'ma' --output_dir 'alger-ia/dziribert-ma' 

The results for our (Masakhane team) can be found in :
[TaskA](TaskA/results-taskA)

[TaskB](TaskB/results-taskB)


