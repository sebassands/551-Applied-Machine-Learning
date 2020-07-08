from pprint import pprint
from time import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_files


data = load_files("aclimdbdata/train/")


pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('clf', AdaBoostClassifier(n_estimators=1000)),
])


parameters = {
    #'clf__n_estimators': (1000),
    #'clf__base_estimator': (DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)),
    # 'clf__learning_rate': (0.5, 1),
    # 'clf__algorithm': ('SAMME.R'),
    #  'vect__max_df': (0.5),
    #'vect__max_features': (None, 5000, 10000, 50000),
    #'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),

}

if __name__ == "__main__":
  
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
