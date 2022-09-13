'''
Cleaning the tabular dataset.
@author:    Behzad 
@date:      10 September 2022
'''
import pandas as pd
import numpy as np
import plotly.express as px
import missingno as msno
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
class CleanData:

    '''
    This class will clean text data and has the following methods:

    csv_to_df(self)
    clean_text_data(self)
    '''

    def csv_to_df(self):

        '''
        Opens csv file in a DataFrame and pre-cleans the data

        RETURNS
        -------
        DataFrame
            A pre cleaned DataFrame
        '''
        df = pd.read_csv("/Users/behzad/AiCore/Facebook_Marketplace_RRS/raw_data/products.csv", lineterminator="\n")
        df = df.drop(df.columns[[0]], axis=1)
        df['price'] = df['price'].str.replace('£', '')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df_pre_clean = df[df.location.notnull()].reset_index() # Only 'price' has null values
        #print(df_pre_clean.columns)
        #print(df_pre_clean.head(10))
        return df_pre_clean

    def clean_text_data(self):

        '''
        Cleans data by removing not usful data from each column
        '''
        df = self.csv_to_df()
        df_tmp_name = df.join(df['product_name'].str.split('|', 0, expand=True).rename(columns={0:'product_name_edited'}))
        df_tmp_ctg = df['category'].str.split('/', expand=True)
        df_tmp_ctg = df_tmp_ctg.rename(columns={0:'category_edited', 1:'category_description'})
        df_tmp_ctg['category_description_edited'] = df_tmp_ctg[df_tmp_ctg.columns[1:]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
        df_ctg = df_tmp_ctg.filter(['category_edited', 'category_description_edited'])
        df_tmp = pd.concat([df_tmp_name, df_ctg], axis=1)
        df_clean = df_tmp.filter(['id', 'product_name_edited', 'category_edited', 'category_description_edited', 'product_description', 'price', 'location'])
        print(df_clean.head(10))

    def text_to_numerical(self, column: str):
        vect = CountVectorizer()
        text = self.df[column].to_list()
        vect.fit(text)
        train = vect.transform(text)
        train.toarray()
        data = pd.DataFrame(train.toarray(), columns=vect.get_feature_names())
        return data

if __name__ == '__main__':
    cln = CleanData() 
    cln.clean_text_data()