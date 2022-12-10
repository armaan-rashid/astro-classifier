import torch
import pandas as pd
import numpy as np
import preprocess
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer


def runLogRegCV(X_train, y_train, X_test, y_test):
    """
    This func implements a sklearn pipeline to take in chart data, 
    with bios as the last column, and output it as a merged vector of just numbers,
    with the text data in the bios already processed. Uses TF-IDF Vectorizer to 
    vectorize natural language data.
    
    Args:
        (DataFrames) X_train, y_train, X_test, y_test: features and labels as dataframes
    Returns:
        (float) accuracy: accuracy of model on the test set
    """
    def init_pipeline():
        """
        Initialize the pipeline with numeric and text transform as appropriate.
        """
        nums_transform = FunctionTransformer(preprocess.extract_chart)
        text_transform = FunctionTransformer(preprocess.extract_bios)
        pipe = Pipeline([
        ('features', FeatureUnion([
                ('numeric_features', Pipeline([
                    ('selector', nums_transform)
                ])),
                ('text_features', Pipeline([
                    ('selector', text_transform),
                    ('vec', TfidfVectorizer(decode_error='ignore', analyzer='word'))
                ]))
            ])),
        ('clf', LogisticRegressionCV())
        ])
        return pipe

    pipe = init_pipeline()
    pipe.fit(X_train, y_train)
    predict_y = pipe.predict(X_test)
    if type(y_test) is not np.ndarray:
        y_test = np.squeeze(np.array(y_test))
    successes, total = np.sum(y_test == predict_y), y_test.shape[0]
    return successes / total
    



def main():
    X_train, y_train, X_test, y_test = preprocess.load_csv_files('train_data.csv', 'train_labels.csv', 'test_data.csv', 'test_labels.csv')
    accuracy = runLogRegCV(X_train, y_train, X_test, y_test)
    print(f'Test accuracy of cross-validated logistic regression is {accuracy}')



if __name__ == '__main__':
    main()