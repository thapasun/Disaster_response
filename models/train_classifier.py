import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support,classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import FeatureUnion
import pickle
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('DisasterResponse', con =engine)

    X = df['message']
    Y = df[df.columns[4:]]
    category_names = Y.columns

    return X,Y,category_names


def tokenize(text):

    text = re.sub(r"[^a-zA-Z0-9]"," ",text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip().lower()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    pipeline = Pipeline([
                ('vect',CountVectorizer(tokenizer = tokenize)),
                ('tfidf',TfidfTransformer()),
                ('clf',MultiOutputClassifier(RandomForestClassifier()))
                ])

    parameters = {'clf__estimator__max_depth':[10,40,None],
                  'clf__estimator__min_samples_leaf':[2,6,12]}
    model = GridSearchCV(pipeline, parameters,n_jobs=4,verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    for label in range(0,len(category_names)):
        print('Feature:',category_names[label])
        print(classification_report(Y_test.iloc[:,label],np.array(Y_pred[:,label])))



def save_model(model, model_filepath):

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
