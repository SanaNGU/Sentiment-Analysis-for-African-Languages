
def main():
    
    import argparse
    import torch
    import wandb
    from torch import nn
    from transformers import AdamW,AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,get_scheduler
    from transformers import get_scheduler,get_linear_schedule_with_warmup,TrainerCallback,AutoConfig,EvalPrediction
    from torch.utils.data import DataLoader
    from datasets import load_dataset, Dataset, DatasetDict,ClassLabel
    from transformers import DataCollatorWithPadding
    import numpy as np
    import evaluate
    do_eval=False
    do_train=True
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from sklearn.model_selection import train_test_split
    import pandas as pd
                    
    
    parser = argparse.ArgumentParser(description='afrisenti')
    parser.add_argument('--language', type=str, default='ha', help='languages: ha,am,dz, ha ,ig ,kr, ma ,pcm ,pt ,sw ,ts ,twi,yo')    
    parser.add_argument('--model_name', type=str, default= 'castorini/afriberta_large', help='name of the deep model')     
    parser.add_argument('--output_dir', type=str, default='castorini/afriberta_large-ha', help='path to save model')      
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')  
    parser.add_argument('--epoch', type=float, default=5, help='number of epochs')
    parser.add_argument('--multilingual', type=bool, default=False, help='Do multilingual')     
    parser.add_argument('--tagged', type=bool, default=False, help='add language tag')     


    args = parser.parse_args()
    checkpoint=args.model_name   #'castorini/afriberta_large'##xlm-roberta-base ##xlm-roberta-large # castorini/afriberta_large #Davlan/naija-twitter-sentiment-afriberta-large#Davlan/xlm-roberta-base- 


    #data_dir= '/home/sanala/Juputer try/afrisent-semeval-2023/modified-training/SubtaskA/'  
    data_dir= '/home/sanala/Juputer try/afrisenti/TaskB/' 
    
    
    if not do_eval:
        train = pd.read_csv(data_dir+'multilingual_train.tsv', delimiter="\t")
        train = train.dropna()
        label_list = train['label'].unique().tolist()
        num_labels = len(label_list)

        train = Dataset.from_pandas(train)
        dataset = DatasetDict(
            {
                "train": train
            }
        )
    else:
        train = pd.read_csv(data_dir+'multilingual_train_new.tsv', delimiter="\t")
        valid = pd.read_csv(data_dir+'multilingual_dev_new.tsv', delimiter="\t")
        valid = valid.dropna()
        label_list = train['label'].unique().tolist()
        num_labels = len(label_list)

        train = Dataset.from_pandas(train)
        valid = Dataset.from_pandas(valid)
        dataset = DatasetDict({"train": train,"validation": valid })
    config = AutoConfig.from_pretrained(checkpoint,num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint,config=config).to(device)

    label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    def preprocess_function(examples):
        # Tokenize the texts
        #print
        texts =(examples['tweet'],)
        result =  tokenizer(*texts, padding='max_length', max_length=200, truncation=True)
        #print(examples['text'])
        #result = tokenizer(examples['text'], examples['text'], padding=padding, max_length=data_args.max_seq_length, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
             result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    
 #####multilingual part   
    langtags = dict()
    if args.multilingual and args.tagged:
        # if you wanna train and do multilingual training with tags then we need to add the tags to the vocab
        langtags = {'am':'<am>', 'dz':'<dz>', 'ha':'<ha>', 'ig':'<ig>', 'kr':'<kr>', 'ma':'<ma>', 'pcm':'<pcm>', 'pt':'<pt>', 'sw':'<sw>', 'ts':'<ts>', 'twi':'<twi>', 'yo':'<yo>'}
        tags = [item for item in langtags.items()]
        
        def add_prefix(example):
            example['tweet'] =  langtags[example['tag']] + example['tweet']
            return example
        
        if do_train:
            tokenizer.add_tokens(tags)
            model.resize_token_embeddings(len(tokenizer))
            train_dataset = dataset['train'].map(add_prefix)

        if do_eval:
            eval_dataset = dataset['valid'].map(add_prefix)
            
        #if training_args.do_predict:
            #predict_dataset = predict_dataset.map(add_prefix)

    metric=evaluate.load("f1")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids,average='macro')

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get('logits')
            # compute custom loss
            loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.242,0.757]).to(device))#0.242,0.757
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    class CustomCallback(TrainerCallback):
    
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
    
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
                return control_copy




    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    results = []
    logging_steps=100
    batch_size=32

 
    #for i in range (len(data['train'])):
    for i in range (1):
        train_data=dataset['train']
        tokenized_datasets_train = train_data.map(preprocess_function,batched=True)
         
        batch_size=32
        logging_steps=100
        training_arg=TrainingArguments('trainerfile',num_train_epochs=args.epoch,logging_steps=logging_steps,learning_rate=args.learning_rate,
                                       per_device_train_batch_size=batch_size)#,per_device_eval_batch_size=batch_size,evaluation_strategy="epoch") 
    
        if do_eval:   
            valid_data=dataset['validation']
            tokenized_datasets_valid = valid_data.map(preprocess_function,batched=True)       
            training_arg=TrainingArguments('trainerfile',num_train_epochs=args.epoch,
                                           logging_steps=logging_steps,learning_rate=args.learning_rate,
                                           per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,evaluation_strategy="epoch") 

  

        trainer=Trainer(model,args=training_arg,train_dataset=tokenized_datasets_train,
                        eval_dataset=tokenized_datasets_valid if do_eval else None,
                        data_collator=data_collator,tokenizer=tokenizer,
                        compute_metrics=compute_metrics)
        result=trainer.train()

    
    
        #trainer=Trainer(model,args=training_arg,train_dataset=tokenized_datasets_train,eval_dataset=tokenized_datasets_valid,data_collator=data_collator,tokenizer=tokenizer,compute_metrics=compute_metrics)
        #optim_scheduler = create_optimizer_and_scheduler(trainer.model) 
        #trainer.optimizer = optim_scheduler
        #trainer.add_callback(CustomCallback(trainer)) 
        results.append(result)

    trainer.save_model(args.output_dir)
    metrics = result.metrics


if __name__ == "__main__":
    main()
    
    '''
CUDA_VISIBLE_DEVICES=3 python run_train.py \
!python run_train_taskB.py  --model_name  'setu4993/LaBSE'  --learning_rate 5e-5 --epoch 2.0 --output_dir 'setu4993/LaBSE-multilingual'  
'''
    
