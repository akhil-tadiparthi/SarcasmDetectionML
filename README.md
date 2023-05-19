# English Sarcasm Detection using the RoBERTa Model

I chose to do the Task A from the [iSarcasamEval](https://github.com/iabufarha/iSarcasmEval). 

## Task Details
SubTask A: Given a text, determine whether it is sarcastic or non-sarcastic;

## How to use the code:
1. To run the code, follow these steps:
* Use Google Colab to run the script and the file [RoBERTa_Code_Submission.ipynb](https://github.com/SnehaYendluri/NLP_project/blob/main/RoBERTa_Code_Submission.ipynb)
* Use the training_data.csv which is from sem2018(train+test) and train 2022 which is augmented. Using this data helped us achieve a good f score 
* Download the train.En.csv for the train data from the [iSarcasamEval Task A Train Data](https://github.com/iabufarha/iSarcasmEval/tree/main/train) and task_A_En_test.csv for the test data from the [iSarcasamEval Task A Test Data](https://github.com/iabufarha/iSarcasmEval/tree/main/test) repository. Move these files into a folder called sarcasm. Then zip the file
* Or download the sarcasm.zip foler and add it to the google collab
* Or download the folder called sarcasm and zip it 
* The following packages were used in our experiments:
```
import matplotlib.pyplot as plt   # Import the matplotlib.pyplot library for data visualization
import seaborn as sns              # Import the seaborn library for data visualization
import pandas as pd                # Import the pandas library for data manipulation and analysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Import scikit-learn metrics
!pip install transformers        # Install the transformers package
!pip install -U torchtext==0.6.0 # Install a specific version of the torchtext package
!pip install pytorch-pretrained-bert pytorch-nlp # Install the pytorch-pretrained-bert and pytorch-nlp packages
import torch                      # Import PyTorch library
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup  # Import the necessary classes from the transformers package
import warnings                   # Import the warnings library to handle warning messages
warnings.filterwarnings('ignore') # Ignore warning messages
import logging                    # Import the logging library to handle log messages
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Ignore log messages related to tokenization_utils_base
from google.colab import drive    # Import the drive module from the google.colab library and mount the Google Drive
drive.mount('/content/drive')
```

2. Data Cleaning
* Upload the test and train data. The train data should have Sarcastic and not sarcastic data. 
* Perform data cleaning to remove any unwanted characters from the text data defined under the function.
```
clean_data
```
  * Save preprocessed data, cropped to max length of the model. Use the following code snippet: 
```
train_df['clean'] = train_df['clean'].apply(lambda x: " ".join(x.split()[:512]))
train_df.to_csv("prep_news.csv")
```

3. ROBERTA Input Formatting 
* We used RoBERTa model, which requires specific formatted inputs. For each tokenized input sentence, we need to intialize a tokenizer and create iterators for the train, validation, and test sets. 
* Use the following helper functions:
  * __save_checkpoint(path, model, valid_loss)__: Save model's state dictionary and valid_loss value to file at specified path
  * __load_checkpoint(path, model)__: Load state dictionary from file at specified path to model
  * __save_metrics(path, train_loss_list, valid_loss_list, global_steps_list)__: Save training loss list, validation loss list, and global steps list to file at specified path
  * __load_metrics(path)__: Load training loss list, validation loss list, and global steps list from file at specified path
* Use a ROBERA Classifier which includes the forward function 
* We pretrain the model and then train the model

6. Model training
* We are using ROBERTA model to train the data
* Use the defined function called evaluate that returns the predicted labels as a list 
  * Use the following code snippet to creat an instance of the ROBERTAClassifier mode: 
 ```
model = ROBERTAClassifier()
model = model.to(device)
load_checkpoint('model.pkl', model)
y_pred = evaluate(model, test_iter)
```
* Here, we are loading BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.


## Metrics

For all the sub-tasks, precision, recall, accuracy and macro-F1 will be reported. The main metrics are:

* SubTask A: F1-score for the sarcastic class. This metric should not be confused with the regular macro-F1. Please use the following code snippet:
```
from sklearn.metrics import f1_score, precision_score, recall_score
f1_sarcastic = f1_score(test_data["sarcastic"], test_data["pred"],average = "binary", pos_label = 1)
print('The final F1 score: ', f1_sarcastic) # returns the f score
```

* Finally we put the final output in the same format as the test file, using the code below

```
result = test_data[['text','pred']]
result.to_csv('results')
```

We got an F Score of 0.43 which is above the threashold of 0.4
