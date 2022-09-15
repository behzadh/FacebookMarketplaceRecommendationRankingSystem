from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import sklearn.metrics as metrics
from nltk.corpus import stopwords

class FbMarketingData:

    '''
    '''
    def __init__(self) -> None:
        self.url_regex = r'http\S+'
        self.path = '/Users/behzad/AiCore/Facebook_Marketplace_RRS/raw_data/'
        self.stop_words = stopwords.words('english')
        self.test_to_train_ratio = 0.1
        self.seed = 37

    def load_data(self, no_null: bool = True):

        '''
        '''
        df = pd.read_csv(self.path + "products_clean.csv", lineterminator='\n')
        df = df[df.price.notnull()].reset_index()
        df['product_description_wo_stopwords'] = df['product_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop_words)]))

        X = df[['product_name_edited', 'product_description_wo_stopwords','location']]
        y = df.price
        return X, y

    def model_data(self):

        '''
        '''
        X, y = self.load_data(True)

        vect = CountVectorizer(ngram_range=(1,3))
        train = sp.hstack(X.apply(lambda col: vect.fit_transform(col)))

        X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = self.test_to_train_ratio, random_state = self.seed)

        clf = LinearRegression()
        clf.fit(X_train, y_train)

        print('Score: ', clf.score(X_test, y_test))

    def pipeline_model_data(self):

        '''
        '''
        X, y = self.load_data(True)
        #vect = CountVectorizer(ngram_range=(1,3))
        preprocess = ColumnTransformer(
            [
                ('vect1', CountVectorizer(ngram_range=(1,3)), 'product_name_edited'),
                ('vect2', CountVectorizer(ngram_range=(1,2)), 'product_description_wo_stopwords'),
                ('vect3', CountVectorizer(ngram_range=(1,2)), 'location'),
            ],
            remainder='passthrough'
        )
        pipeline = Pipeline(
            [
                ("vect", preprocess),
                ('tfidf', TfidfTransformer()),               
                ("clf", LinearRegression())
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_to_train_ratio, random_state = self.seed)

        pipeline.fit(X_train, y_train)

        print('Score: ', pipeline.score(X_test, y_test))

    def main(self):
        #self.model_data()
        self.pipeline_model_data()