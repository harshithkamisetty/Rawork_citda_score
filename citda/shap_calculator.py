import pandas as pd
import numpy as np
import torch
import transformers
import datasets
import shap
import pickle
import json

class ShapCalculator:
    def __init__(self, data_file, text_column, label_column, shap_values_filename,Features, 
                model_name="nateraw/bert-base-uncased-emotion", 
                tokenizer_name="nateraw/bert-base-uncased-emotion",
                ):
        self.shap_values = None
        self.df = pd.read_csv(data_file, nrows=2)
        self.shap_values_filename = shap_values_filename
        self.text_column=text_column
        self.label_column=label_column
        print("Reading data...")
        print("DataFrame's shape:", self.df.shape)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using device:", self.device)

        print("Loading model:", model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        print("Loading tokenizer:", tokenizer_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.labels = sorted(self.model.config.label2id, key=self.model.config.label2id.get)
        
        self.pred = transformers.pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0, top_k=None)
        self.Features= Features

    def calculate(self):
        #self.shap_values_filename = shap_values_filename
     
        # tokenizer.to(device)
        
        
        print("Calculating SHAP values...")
        explainer = shap.Explainer(self.pred, self.tokenizer, output_names=self.labels)
        self.shap_values = explainer(self.df[self.text_column][:])
        
        # Saving the shap values to a file
        pickle.dump(self.shap_values, open(self.shap_values_filename, 'wb'))
        print(f"SHAP values saved to {self.shap_values_filename}.")  

    def load_shap_values(self):
        print("Loading the Sharp Values")
        self.shap_values = pickle.load(open(self.shap_values_filename, 'rb'))
        return self.shap_values

    def tokenization(self, text):
        # tokenize the text column
        tokenized = self.tokenizer.encode(text, add_special_tokens=True)

        # decoded = tokenized.apply(lambda x: tokenizer.decode(x))
        # decode each integer in the tokenized input separately
        text = []
        for tokens in tokenized:
            token = self.tokenizer.convert_ids_to_tokens([tokens])[0]
            text.append(token)
        #print("Tokenized:", tokenized) # the first row's encoded tokens
        #print("Decoded:",', '.join(text)) # the firs row's decoded tokens
        return ', '.join(text)
    
    def predict(self, texts):
        preds = self.pred(texts)
        return preds


    def liwc_score(self,idx, exclude_cols=['idx','text','label','Segment','WC']):
        nonzero_indices=[]
        nonzero_values=[]
        temp_df=self.df.drop(columns=exclude_cols)

        row = temp_df.iloc[idx]
        non_zero_indexes = np.nonzero(row.values)[0]
        non_zero_values = row.iloc[non_zero_indexes]

        # Display the indexes and values of non-zero elements
        print("Non-zero elements:")
        for idx, val in enumerate(non_zero_values):
            nonzero_indices.append(non_zero_indexes[idx])
            nonzero_values.append(val)
        return nonzero_indices,nonzero_values


    def create_liwc_shap(self, filename = 'liwc_shap.json'):
        #check if shap values are loaded or not
        if self.shap_values == None:
            self.load_shap_values()      
        output=[]
        for idx,row in self.df.iterrows():
            tokens=self.tokenization(row[self.text_column])
            predict_result=self.predict(row[self.text_column])
            max_pair = max(predict_result[0], key=lambda pair: pair['score'])
            pred_index=self.Features.index(max_pair['label'])
            specfic_shap_values=self.shap_values[idx]
            out_shap_values=[]
            
            for item in specfic_shap_values.values:
                out_shap_values.append(item[pred_index])
            
            liwc_indices, liwc_values = self.liwc_score(idx)
            #print(liwc_indices, liwc_values)
            
            row_data = {"index": idx, 
                        "actual_label": row['emotion'],
                        "pred_label": self.Features.index(max_pair['label']),
                        "pred_score": max_pair['score'],
                        "shap_values":str(out_shap_values),
                        "tokens":tokens,
                        "liwc_index":str(liwc_indices),
                        "liwc_value":str(liwc_values)}
            output.append(row_data)
        
        # Saving the dictionary to a json file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"LIWC shap values saved to {filename}.")
