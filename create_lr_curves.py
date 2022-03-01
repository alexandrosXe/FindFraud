import pandas as pd
import numpy as np
from ast import literal_eval
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
# np.random.seed(seed=0)
# import tensorflow
# from tensorflow.random import set_seed
# set_seed(0)
from tqdm import tqdm


%%capture
#download data
!gdown --id 1ferhQgQJH0CIzoPkRC_orqpRMO7FJB3e
!gdown --id 1Rs9ZL7AGJivPaf2Agy8QDL76YHG6lMB1

api = pd.read_excel("final_dataset.xlsx")
print("dataset shape: ", api.shape)
action_vocab = pd.read_excel("s_names.xlsx")

# one line is broken, throw it away
broken_times = api[api.times.apply(lambda x: x[-1]!="]")]
assert broken_times.shape[0] == 1 # one line, right?
assert broken_times.iloc[0].is_fraud==0 # and not a fraud, right?

# ignore the broken line
api = api[api.times.apply(lambda x: x[-1]=="]")]

#build the raw text, using the names and the (index-inverted) tokens 
action_names = action_vocab.Name.to_list()
id_to_action = {str(i):a for i,a in enumerate(action_names)}
action_to_id = {a:str(i) for i,a in enumerate(action_names)}

# T9: map the actions to proper names and tokenise 

# Recall to cast the strings into lists
api.actions = api.actions.apply(literal_eval)

api["times"] = api.times.apply(literal_eval).apply(lambda x: [i/1000 for i in x]) # in seconds
api["Action time mean"] = api.times.apply(np.mean) # mean time elapsed from action to action
api["Action time std"] = api.times.apply(np.std) # st. dev of time from action to action
api["log(amount)"] = api.Amount.apply(np.log); # outliers may be important (x->logx)
api["Transaction Type"] = api.is_fraud.apply(lambda x: "Fraud" if x else "Non Fraud")
api["time_to_first_action"] = api.times.apply(lambda x: x[1] if len(x)>1 else 0)
api["actions_str"] = api.actions.apply(lambda x: " ".join([id_to_action[str(i[0])] for i in x if len(i)>0]))
api.head(2)


def fine_tune_threshold(y_true, preds, method = 'f_05'):
  f_scores = []
  ths = []
  for th in range(0,100,1):
    th = th/100
    rounded_preds = pd.Series(preds)
    rounded_preds = rounded_preds.apply(lambda x: 1 if x > th else 0)
    if method == 'f_05':
      f_scores.append(fbeta_score(y_true, rounded_preds, beta = 0.5))
    elif method == 'f_2':
      f_scores.append(fbeta_score(y_true, rounded_preds, beta = 2))
    elif method == 'f_1':
      f_scores.append(f1_score(y_true, rounded_preds))
    else:
      assert "Wrong Metric"
    ths.append(th)
  max_index = np.nanargmax(f_scores)
  best_th = ths[max_index]
  print("best thershold which max f_score is: ", best_th, " with "+ method + "score: ", f_scores[max_index])
  #print(f_scores)
  return best_th

def save_train_predictions(mc_gold, mc_preds, f05_preds, f1_preds, f2_preds, features = "Ml", ratios = None):
  # AUC and AUPRC
  models = []
  auprs = []
  roc_aucs = []
  f05s = []
  f1s = []
  f2s = []
  auprs_sems = []
  roc_aucs_sems = []
  f05s_sems = []
  f1s_sems = []
  f2s_sems = []
  for k in mc_preds:
    models.append(k)
    aupr = [average_precision_score(g,p) for g,p in zip(mc_gold[k],mc_preds[k])]
    auprs.append(list(np.round(aupr, 3)))
    #auprs_sems.append(np.round(aupr.sem(), 3))

    roc_auc = [roc_auc_score(g,p) for g,p in zip(mc_gold[k],mc_preds[k])]
    roc_aucs.append(list(np.round(roc_auc, 3)))
    #roc_aucs_sems.append(np.round(roc_auc.sem(), 3))

    f05 = [fbeta_score(g,p, beta=0.5) for g,p in zip(mc_gold[k], f05_preds[k])]
    f05s.append(list(np.round(f05, 3)))
    #f05s_sems.append(np.round(f05, 3))

    
    f1 = [f1_score(g,p) for g,p in zip(mc_gold[k], f1_preds[k])]
    f1s.append(list(np.round(f1, 3)))
    #f1s_sems.append(np.round(f1.sem(), 3))

    
    f2 = [fbeta_score(g,p, beta=2) for g,p in zip(mc_gold[k], f2_preds[k])]
    f2s.append(list(np.round(f2, 3)))
    #f2s_sems.append(np.round(f2.sem(), 3))

  ratios_all  = [ratios for i in range(len(models))]

  #save predictions in proper format
  for k in mc_preds:
    preds = []
    for arr in mc_preds[k]:
      preds.append(list(arr))
    mc_preds[k] = preds
  preds = [mc_preds[k] for k in mc_preds]
  results_df = pd.DataFrame({'Model':models, 'Train_AUPRC':auprs, 'Train_ROC_AUC':roc_aucs, 'Train_F05':f05s, 'Train_F1':f1s, 'Train_F2':f2s, 
                              'Features': [features for i in range(len(models))], 'Train_Preds' : preds, 'Ratios' : ratios_all})
  #results_df.to_csv(path, index = False)
  return results_df

def save_test_predictions(mc_gold, mc_preds, f05_preds, f1_preds, f2_preds, features = "Ml"):
  # AUC and AUPRC
  models = []
  auprs = []
  roc_aucs = []
  f05s = []
  f1s = []
  f2s = []
  auprs_sems = []
  roc_aucs_sems = []
  f05s_sems = []
  f1s_sems = []
  f2s_sems = []
  for k in mc_preds:
    models.append(k)
    aupr = [average_precision_score(g,p) for g,p in zip(mc_gold[k],mc_preds[k])]
    auprs.append(list(np.round(aupr, 3)))
    #auprs_sems.append(np.round(aupr.sem(), 3))

    roc_auc = [roc_auc_score(g,p) for g,p in zip(mc_gold[k],mc_preds[k])]
    roc_aucs.append(list(np.round(roc_auc, 3)))
    #roc_aucs_sems.append(np.round(roc_auc.sem(), 3))

    f05 = [fbeta_score(g,p, beta=0.5) for g,p in zip(mc_gold[k], f05_preds[k])]
    f05s.append(list(np.round(f05, 3)))
    #f05s_sems.append(np.round(f05, 3))

    
    f1 = [f1_score(g,p) for g,p in zip(mc_gold[k], f1_preds[k])]
    f1s.append(list(np.round(f1, 3)))
    #f1s_sems.append(np.round(f1.sem(), 3))

    
    f2 = [fbeta_score(g,p, beta=2) for g,p in zip(mc_gold[k], f2_preds[k])]
    f2s.append(list(np.round(f2, 3)))
    #f2s_sems.append(np.round(f2.sem(), 3))

  ratios_all  = [ratios for i in range(len(models))]

  #save predictions in proper format
  for k in mc_preds:
    preds = []
    for arr in mc_preds[k]:
      preds.append(list(arr))
    mc_preds[k] = preds
  preds = [mc_preds[k] for k in mc_preds]
  results_df = pd.DataFrame({'Model':models, 'Test_AUPRC':auprs, 'Test_ROC_AUC':roc_aucs, 'Test_F05':f05s, 'Test_F1':f1s, 'Test_F2':f2s, 'Test_Preds' : preds})
  #results_df.to_csv(path, index = False)
  return results_df

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes 
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from scipy.sparse import hstack


#5-Fold Monte carlo cross validation 

# mc_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : [],  'MLP' : []}
# f05_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : [],  'MLP' : []}
# f1_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : [],  'MLP' : []}
# f2_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : [],  'MLP' : []}
# mc_gold = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : [], 'MLP' : []}

mc_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
f05_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
f1_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
f2_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
mc_gold = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}

train_mc_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
train_f05_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
train_f1_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
train_f2_predictions = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}
train_mc_gold = {'LR' : [], 'RF' : [], 'KNN' : [], 'SVM' : []}



X, y = api[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq', 'actions_str', 'is_fraud']], api.is_fraud

# X = X[:4000]
# y = y[:4000]

mc_splits = 1

ratios = []

for i in range(mc_splits):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = i)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = i)

  print("\nX_train shape: ", X_train.shape[0])
  print("X_val shape: ",X_val.shape[0])
  print("X_test shape: ",X_test.shape[0])
  # print("Y train label dist: ",  y_train.value_counts())
  # print("Y test label dist: ",y_test.value_counts())

batch_size = 5000

train_frauds = X_train[X_train.is_fraud == 1].reset_index(drop = True)
y_frauds = train_frauds.is_fraud

train_non_frauds = X_train[X_train.is_fraud == 0].reset_index(drop = True)
#y_non_frauds = train_non_frauds.is_fraud


# do the rest batches
for i in tqdm(range((train_non_frauds.shape[0]//batch_size)+1)):
  train_instances = train_non_frauds[:batch_size*(i+1)]
  y_train_instances = train_instances.is_fraud

  #concat frauds with non frauds
  train_instances = pd.concat([train_instances, train_frauds]).sample(frac = 1).reset_index(drop = True)
  y_train_instances = pd.concat([y_train_instances, y_frauds]).sample(frac = 1).reset_index(drop = True)

  ratios.append(y_frauds.shape[0]/y_train_instances.shape[0]*100)

  #extract tf*idf features
  vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.2)
  X_train_tf_idf = vec.fit_transform(train_instances.actions_str)
  X_val_tf_idf = vec.transform(X_val.actions_str)
  X_test_tf_idf = vec.transform(X_test.actions_str)

  #ml features 
  X_train_ml = train_instances[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]
  X_val_ml = X_val[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]
  X_test_ml = X_test[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]


  #concatenate ml features with tf-idf features
  X_train_features = hstack((X_train_tf_idf, X_train_ml.to_numpy()))
  X_val_features = hstack((X_val_tf_idf, X_val_ml.to_numpy()))
  X_test_features = hstack((X_test_tf_idf, X_test_ml.to_numpy()))

  lr = LogisticRegression().fit(X_train_features, y_train_instances)
  rf = RandomForestClassifier().fit(X_train_features,  y_train_instances)
  #knn = KNeighborsClassifier(n_neighbors=2).fit(X_train_features,  y_train_instances)
  #svm_clf = svm.SVC(probability=True).fit(X_train_features,  y_train_instances)
  #gb_clf = GradientBoostingClassifier().fit(X_train, y_train)
  #mlp_clf = MLPClassifier(hidden_layer_sizes = (100,), activation = 'relu',solver = 'adam', batch_size = 128, max_iter=200).fit(X_train, y_train)

  #make preds 
  for k, clf in (("LR",lr), ("RF",rf)): #, ("KNN",knn), ("SVM",svm_clf)): #, ("MLP",mlp_clf)):
  #for k, clf in (("LR",lr), ("SVM",svm_clf)):

    #preds on val set
    val_preds = clf.predict_proba(X_val_features)[:,1]
    test_preds = pd.Series(clf.predict_proba(X_test_features)[:,1])
    train_preds = pd.Series(clf.predict_proba(X_train_features)[:,1])

    print(val_preds.shape)

    #train preds

    #test preds 

    best_th = fine_tune_threshold(y_val, val_preds, method='f_05')
    train_f05_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f05_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))

    best_th = fine_tune_threshold(y_val, val_preds, method='f_1')
    train_f1_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f1_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))
  
    best_th = fine_tune_threshold(y_val, val_preds, method='f_2')
    train_f2_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f2_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))

    train_mc_predictions[k].append(clf.predict_proba(X_train_features)[:,1])
    mc_predictions[k].append(clf.predict_proba(X_test_features)[:,1])

    train_mc_gold[k].append(y_train_instances)
    mc_gold[k].append(y_test)


### DO THE LAST BATCH
if (train_non_frauds.shape[0]%batch_size)!=0:
  train_instances = train_non_frauds
  y_train_instances =  train_instances.is_fraud

  #concat frauds with non frauds
  train_instances = pd.concat([train_instances, train_frauds]).sample(frac = 1).reset_index(drop = True)
  y_train_instances = pd.concat([y_train_instances, y_frauds]).sample(frac = 1).reset_index(drop = True)

  ratios.append(y_frauds.shape[0]/y_train_instances.shape[0]*100)

  #extract tf*idf features
  vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.2)
  X_train_tf_idf = vec.fit_transform(train_instances.actions_str)
  X_val_tf_idf = vec.transform(X_val.actions_str)
  X_test_tf_idf = vec.transform(X_test.actions_str)

  #ml features 
  X_train_ml = train_instances[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]
  X_val_ml = X_val[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]
  X_test_ml = X_test[['execution_time', 'log(amount)', 'device_freq','ip_freq', 'beneficiary_freq', 'application_freq']]


  #concatenate ml features with tf-idf features
  X_train_features = hstack((X_train_tf_idf, X_train_ml.to_numpy()))
  X_val_features = hstack((X_val_tf_idf, X_val_ml.to_numpy()))
  X_test_features = hstack((X_test_tf_idf, X_test_ml.to_numpy()))

  lr = LogisticRegression().fit(X_train_features,  y_train_instances)
  rf = RandomForestClassifier().fit(X_train_features,  y_train_instances)
  # knn = KNeighborsClassifier(n_neighbors=2).fit(X_train_features,  y_train_instances)
  # svm_clf = svm.SVC(probability=True).fit(X_train_features,  y_train_instances)
  #gb_clf = GradientBoostingClassifier().fit(X_train, y_train)
  #mlp_clf = MLPClassifier(hidden_layer_sizes = (100,), activation = 'relu',solver = 'adam', batch_size = 128, max_iter=200).fit(X_train, y_train)

   #make preds 
  for k, clf in (("LR",lr), ("RF",rf)):#, ("KNN",knn), ("SVM",svm_clf)): #, ("MLP",mlp_clf)):
  #for k, clf in (("LR",lr), ("SVM",svm_clf)):

    #preds on val set
    val_preds = clf.predict_proba(X_val_features)[:,1]
    test_preds = pd.Series(clf.predict_proba(X_test_features)[:,1])
    train_preds = pd.Series(clf.predict_proba(X_train_features)[:,1])

    print(val_preds.shape)

    #train preds

    #test preds 

    best_th = fine_tune_threshold(y_val, val_preds, method='f_05')
    train_f05_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f05_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))

    best_th = fine_tune_threshold(y_val, val_preds, method='f_1')
    train_f1_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f1_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))
  
    best_th = fine_tune_threshold(y_val, val_preds, method='f_2')
    train_f2_predictions[k].append(train_preds.apply(lambda x: 1 if x > best_th else 0))
    f2_predictions[k].append(test_preds.apply(lambda x: 1 if x > best_th else 0))

    train_mc_predictions[k].append(clf.predict_proba(X_train_features)[:,1])
    mc_predictions[k].append(clf.predict_proba(X_test_features)[:,1])

    train_mc_gold[k].append(y_train_instances)
    mc_gold[k].append(y_test)


  
df_train = save_train_predictions(train_mc_gold, train_mc_predictions, train_f05_predictions, train_f1_predictions, train_f2_predictions, features = "Tf-idf with ML", ratios = ratios)
df_test = save_test_predictions(mc_gold, mc_predictions, f05_predictions, f1_predictions, f2_predictions, features = "Tf-idf with ML")

df = pd.merge(df_train, df_test, on = 'Model')
df.to_csv("drive/MyDrive/Colab Notebooks/Papers/2022/Fraud Detection Using NLP/Learning Curves/tf_idf_with_ml_lr.csv", index = False)
#df = save_predictions("check.csv", mc_gold, mc_predictions, f05_predictions, f1_predictions, f2_predictions, features = "Tf-idf")

df.head(df.shape[0])
