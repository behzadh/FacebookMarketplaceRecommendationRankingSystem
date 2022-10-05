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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.svm import SVC

class FbMarketing:

    '''
    '''
    def __init__(self) -> None:
        self.path = '/Users/behzad/AiCore/Facebook_Marketplace_RRS/raw_data/'
        self.test_to_train_ratio = 0.3
        self.seed = 37

    def load_data(self):

        '''
        '''
        df = pd.read_pickle(self.path + 'products_w_imgs.pkl')
        shuffled = df.sample(frac=1, random_state=37).reset_index()

        X = shuffled['image']
        y = (pd.get_dummies(shuffled['category_edited'])).values
        label = pd.DataFrame(y)
        X_tmp = np.asarray([np.asarray(shuffled['image'][x]) for x in range(shuffled.shape[0])])
        X_tmp=np.reshape(X_tmp,(shuffled.shape[0], 1, 64, 64))
        nsamples, nx, ny, nz = X_tmp.shape

        X = X_tmp.reshape((nsamples,nx*ny*nz))
        y = label.apply(lambda x: x.argmax(), axis=1).values
        return X, y

    def model_data(self):

        '''
        '''
        X, y = self.load_data()

        scaler = StandardScaler()
        X_scaler = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size = self.test_to_train_ratio, random_state = self.seed)
        
        clf = SVC(kernel='linear', probability=True, random_state=37)
        clf.fit(list(X_train), y_train)

        y_pred = clf.predict(X_test)

        # calculate accuracy
        accuracy = accuracy_score(y_pred, y_test)
        print('Model accuracy is: ', accuracy)
        print('Score: ', clf.score(X_test, y_test))

    def main(self):
        self.model_data()
        #self.pipeline_model_data()
        #self.load_data()