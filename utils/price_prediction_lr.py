import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from nltk.corpus import stopwords

class FbMarketingPricePrediction:

    '''
    This Class will perform a linear regression model to predict the price of product base on their name, description and their location 
    and has the following functions:

    __init__(self)
    load_data(self)
    lr_clf(self)
    pipeline_lr_clf(self)
    main(self)
    '''
    def __init__(self) -> None:
        self.url_regex = r'http\S+'
        self.path = '/Users/behzad/AiCore/Facebook_Marketplace_RRS/raw_data/'
        self.stop_words = stopwords.words('english')
        self.test_to_train_ratio = 0.1
        self.seed = 37

    def load_data(self):

        '''
        Loads raw data, remove null values and stop words and, defines the lables and target

        RETURNS
        -------
        X, y
            It returens labels (X) and the target (y)
        '''
        df = pd.read_csv(self.path + "products_clean.csv", lineterminator='\n')
        df = df[df.price.notnull()].reset_index()
        df['product_description_wo_stopwords'] = df['product_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop_words)]))

        X = df[['product_name_edited', 'product_description_wo_stopwords','location']]
        y = df.price
        return X, y

    def lr_clf(self):

        '''
        Builds a simple Linear Regression model to predic the price
        '''
        X, y = self.load_data(True)

        vect = CountVectorizer(ngram_range=(1,3)) # vectorize the test
        train = sp.hstack(X.apply(lambda col: vect.fit_transform(col))) # joins all columns after vectorizing the text

        X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = self.test_to_train_ratio, random_state = self.seed)

        clf = LinearRegression()
        clf.fit(X_train, y_train)
        print('Score: ', clf.score(X_test, y_test))

    def pipeline_lr_clf(self):

        '''
        Creates a Pipeline to perform the Linear Regression
        '''
        X, y = self.load_data(True)
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
        #self.lr_clf()
        self.pipeline_lr_clf()