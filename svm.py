#!/usr/bin/env python

"""
LfD Assignment 1 - Hyperparameter Tuning via Command Line

This script allows the user to run classifiers with custom hyperparameters provided through the command line.

Available classifiers:
- Naive Bayes
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM) (2 implementations)

Available command-line options:
-t                  Define train file. Looks for 'train.tsv' default.
-d                  Define dev file. Looks for 'dev.tsv' default. Also used to define the test file.
-s                  Carry out sentiment analysis. If this is omitted, multiclass analysis will be carried out.
-tf                 Use the TF-IDF vectorizer instead of CountVectorizer
-c                  Choose classifier to use. Options are: 'naive_bayes', 'decision_tree', 'random_forest',
                                            'k_neighbors', 'linear_svc', 'svc', 'all_classifiers'.
-- alpha
--ngram_range       Ngram range (e.g. '1,2' for unigrams and bigrams)
--min_df            Min document frequency (default: 2)
--max_df            Max document frequency (default: 0.9)
--max_features      Max features for the vectorizer (default: None)
--max_depth         Max depth for Decision Tree (default: None)
--min_samples_split Min samples to split (default: 2)
--n_estimators      Number of trees for Random Forest (default: 100)
--n_neighbors       Number of neighbors for K-Neighbors (default: 5)
--C                 C parameter for SVC and Linear SVC (default: 1.0)
--kernel            Kernel for SVC (default: 'rbf')
-p                  Displays a plotted version of the confusion matrix for the report. May not always work
                        from the commandline if display packages are not installed.

Usage example:
python script.py -c naive_bayes --alpha 0.5 --max_features 1000

How to run best model:
python lfd_assignment1.py -c naive_bayes -d test.tsv -tf --alpha 0.5
"""

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


def create_arg_parser():
    """Creates argumentparser and defines command-line options that can be called upon."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train.tsv', type=str, help="Train file (default: train.tsv)")
    parser.add_argument("-d", "--dev_file", default='dev.tsv', type=str, help="Dev/test file (default: test.tsv)")
    parser.add_argument("-s", "--sentiment", action="store_true", help="Perform sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true", help="Use TF-IDF vectorizer (default: CountVectorizer)")
    parser.add_argument("-c", "--classifier", type=str,
                        choices=['naive_bayes', 'decision_tree', 'random_forest', 'k_neighbors', 'linear_svc', 'svc',
                                 'all_classifiers'], help="Classifier to use")
    # MNB hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha for Naive Bayes (default: 1.0)")

    # Vectorizer hyperparameters for features tuning
    parser.add_argument("--ngram_range", type=str, default='1,1', help="Ngram range (e.g. '1,2' for unigrams "
                                                                       "and bigrams)")
    parser.add_argument("--min_df", type=float, default=0.0, help="Min document frequency (default: 2)")
    parser.add_argument("--max_df", type=float, default=1.0, help="Max document frequency (default: 0.9)")
    parser.add_argument("--max_features", type=int, help="Max features for the vectorizer (default: None)")

    # Decsion Tree hyperparameters
    parser.add_argument("--max_depth", type=int, help="Max depth for Decision Tree (default: None)")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples to split (default: 2)")

    # Random Forest hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees for Random Forest (default: 100)")

    # K-Neighbors hyperparameters
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for K-Neighbors (default: 5)")

    # Linear SVC / SVC hyperparameters
    parser.add_argument("--C", type=float, default=1.0, help="C parameter for SVC and Linear SVC (default: 1.0)")
    parser.add_argument("--kernel", type=str, default='rbf', help="Kernel for SVC (default: 'rbf')")
    parser.add_argument("-p", "--plot_show", action="store_true",
                        help="Displays a plotted version of the confusion matrix for the report. May not always work"
                             "from the commandline if display packages are not installed.")

    return parser.parse_args()


def read_corpus(corpus_file, use_sentiment):
    """Reads the corpus file and splits lines into lists of
    documents and corresponding lists of labels. If use_sentiment
    is true, use the binary (sentiment) labels. Else, use the multi-
    class (category) labels."""
    documents, labels = [], []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])
    return documents, labels


def print_evaluation(Y_test, Y_pred):
    """Takes true labels and predicted labels and
    prints evaluation measures (a classification report
    and confusion matrix)"""
    print('\n*** CLASSIFICATION REPORT ***')
    print(classification_report(Y_test, Y_pred))

    labels = ['NOT', 'OFF']
    print('\n*** CONFUSION MATRIX ***')
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    print(' '.join(labels))
    print(cm)
    # If the --plot_show argument is given, show pyplt version of the confusion matrix
    if args.plot_show:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labels)
        disp.plot()
        plt.show()


def train_and_evaluate(vec, X_train, Y_train, X_test, Y_test, classifier, classifier_name, labels):
    """General function that trains and evaluates a classifier with default settings."""
    model = Pipeline([('vec', vec), ('cls', classifier)])
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    print(f'***** {classifier_name.upper()} *****')
    print_evaluation(Y_test, Y_pred)


def train_and_evaluate_with_grid_search(vec, X_train, Y_train, X_test, Y_test, classifier, classifier_name, param_grid,
                                        ):
    """Trains and evaluates using GridSearchCV for hyperparameter tuning."""
    model = Pipeline([('vec', vec), ('cls', classifier)])
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters for {classifier_name}: {grid_search.best_params_}")
    Y_pred = best_model.predict(X_test)

    print(f'***** {classifier_name.upper()} WITH GRID SEARCH *****')
    print_evaluation(Y_test, Y_pred,)



# Define parameter grids for hyperparameter tuning
linear_svc_param_grid = {
    'cls__C': [0.01, 0.1, 1, 10, 100],
    'cls__max_iter': [1000, 5000, 10000]
}

svc_param_grid = {
    'cls__C': [0.1, 1, 10, 100],
    'cls__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'cls__gamma': ['scale', 'auto'],
    'cls__degree': [2, 3, 4]  # Relevant only for 'poly' kernel
}


if __name__ == "__main__":
    # Parse command-line arguments
    args = create_arg_parser()

    # Load training and testing data
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Handle document frequency arguments for vectorizer
    if args.min_df > 1:
        args.min_df = int(args.min_df)
    if args.max_df > 1:
        args.max_df = int(args.max_df)

    # Convert ngram_range from string to tuple
    ngram_range = tuple(map(int, args.ngram_range.split(',')))

    # Choose vectorizer
    vec = (TfidfVectorizer if args.tfidf else CountVectorizer)(
        ngram_range=ngram_range,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        preprocessor=lambda x: x, tokenizer=lambda x: x, token_pattern=None
    )

    # Define classifiers with hyperparameters based on command-line arguments
    classifiers = {
        'naive_bayes': (MultinomialNB(alpha=args.alpha), "Naive Bayes"),
        'decision_tree': (
            DecisionTreeClassifier(max_depth=args.max_depth, min_samples_split=args.min_samples_split),
            "Decision Tree"),
        'random_forest': (RandomForestClassifier(n_estimators=args.n_estimators), "Random Forest"),
        'k_neighbors': (KNeighborsClassifier(n_neighbors=args.n_neighbors), "K Neighbors"),
        'linear_svc': (LinearSVC(), "Linear SVC"),  # Grid search will handle 'C' and 'max_iter'
        'svc': (SVC(), "SVC")  # Grid search will handle 'C', 'kernel', and 'gamma'
    }

    # Determine if we are using grid search or default training
    if args.classifier == 'all_classifiers':
        for clf_key, (clf, clf_name) in classifiers.items():
            train_and_evaluate(vec, X_train, Y_train, X_test, Y_test, clf, clf_name, args.sentiment)
    else:
        clf, clf_name = classifiers[args.classifier]
        train_and_evaluate(vec, X_train, Y_train, X_test, Y_test, clf, clf_name, args.sentiment)