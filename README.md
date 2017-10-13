# Sentiment-Analysis
Classify Sentiment based on movie reviews using Decision Tree, Logistic Regression, Random Forest And Voting Classifiers

1. Extract features.
In the function,
§ Stemming with PorterStemmer 
§ Used unigram
§ Removed stop words
§ Removed words with document frequency lower than 1% or higher than 99% of all
sentences.
2. Since the labelled sentiment is categorical,developed the below classifiers for classification
§ LogisticRegression10
§ DecisionTreeClassifier11 12
§ RandomForestClassifier13
3. Re-trained and tested on dataset with VotingClassifier. Used the previous 3 classifiers as
candidate classifiers.
