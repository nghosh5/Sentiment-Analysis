# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:06:56 2017

@author: Vineet
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.stem.porter import PorterStemmer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier
#importing training and test csvs 
train = pd.read_csv('C:/Users/Vineet/Desktop/Adv Text/Session3/train.csv', encoding='latin1', header=None)
test=pd.read_csv('C:/Users/Vineet/Desktop/Adv Text/Session3/test.csv', encoding='latin1', header=None)
X_train=train[0]
y_train=train[1]
X_test=test[0]
y_test=test[1]
X_combined=[]
X_combined.extend(X_train)
X_combined.extend(X_test)
# TfIDf scores for train data
X_combined = pd.Series.tolist(X_combined)

ps=PorterStemmer()
analyzer = TfidfVectorizer().build_analyzer()


def stemmed_words(doc):
    return (ps.stem(w) for w in analyzer(doc))

vectorizer = TfidfVectorizer(input='content',
                        use_idf=True, # utiliza o idf como peso, fazendo tf*idf
                        norm=None, # normaliza os vetores
                        smooth_idf=False, #soma 1 ao N e ao ni => idf = ln(N+1 / ni+1)
                        sublinear_tf=False, #tf = 1+ln(tf)
                        binary=False,
                        min_df=.01, max_df=.99, max_features=None,
                        lowercase=True,
                        stop_words='english',
                        strip_accents='unicode', # retira os acentos
                        ngram_range=(1,1), preprocessor=None, tokenizer=None, vocabulary=None, analyzer=stemmed_words    
             )
X_combined_tf = vectorizer.fit_transform(X_combined)

#splitting train and test
X_train_tf=X_combined_tf[:6820]
X_test_tf=X_combined_tf[6820:]

#Logistic Regression on Train data

logreg = linear_model.LogisticRegression(C=.001, multi_class='multinomial',solver='lbfgs',random_state=1)
results=logreg.fit(X_train_tf, y_train)   
Y_train_pred = logreg.predict(X_train_tf)
Y_train_pred_prob = logreg.predict_proba(X_train_tf)
print(logreg.score(X_train_tf, y_train)) #39% on train data
print(classification_report(y_train, Y_train_pred))

#Logistic Regression on Test data

Y_test_pred = logreg.predict(X_test_tf)
Y_test_pred_prob = logreg.predict_proba(X_test_tf)
print(logreg.score(X_test_tf, y_test)) #35% on test data
print(classification_report(y_test, Y_test_pred))
filepath='C:/Users/Vineet/Desktop/Adv Text/Session3'

def save_list(result,filepath):
    with open(filepath,'w') as out:
        for x in result:
            out.write(str(x) + '\n')
save_list(Y_test_pred,r'C:/Users/Vineet/Desktop/Adv Text/Session3/664694644_logistic.txt')
#Decision Tree
clf = tree.DecisionTreeClassifier(random_state=1)
results = clf.fit(X_train_tf ,y_train)
Y_train_pred=clf.predict(X_train_tf)
Y_train_pred_prob = clf.predict_proba(X_train_tf)
print(clf.score(X_train_tf, y_train)) #96% on train data
print(classification_report(y_train, Y_train_pred))

#Decision Tree on Test data

Y_test_pred = clf.predict(X_test_tf)
Y_test_pred_prob = clf.predict_proba(X_test_tf)
print(clf.score(X_test_tf, y_test)) #27% on test data
print(classification_report(y_test, Y_test_pred))
save_list(Y_test_pred,r'C:/Users/Vineet/Desktop/Adv Text/Session3/664694644_tree.txt')

#Ramdom Classifier on Train Data


clf1 = RandomForestClassifier(random_state=1)
results = clf1.fit(X_train_tf ,y_train)
Y_train_pred=clf1.predict(X_train_tf)
Y_train_pred_prob = clf1.predict_proba(X_train_tf)
print(clf1.score(X_train_tf, y_train)) #95% on train data
print(classification_report(y_train, Y_train_pred))


#Ramdom Classifier on Test data

Y_test_pred = clf1.predict(X_test_tf)
Y_test_pred_prob = clf1.predict_proba(X_test_tf)
print(clf1.score(X_test_tf, y_test)) #27.6% on test data
print(classification_report(y_test, Y_test_pred))
save_list(Y_test_pred,r'C:/Users/Vineet/Desktop/Adv Text/Session3/664694644_rf.txt')

#Voting Classifier

eclf = VotingClassifier(estimators=[('lr', logreg), ('rf', clf), ('ran', clf1)],
                        voting='hard', weights=[2, 2, 1])

eclf = eclf.fit(X_train_tf, y_train)
y_train_prd = eclf.predict(X_test_tf)
Y_test_pred = clf1.predict(X_test_tf)
print(eclf.score(X_test_tf, y_test)) #0.332358104154
save_list(Y_test_pred,r'C:/Users/Vineet/Desktop/Adv Text/Session3/664694644_voting.txt')
