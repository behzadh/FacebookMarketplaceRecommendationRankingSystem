# Facebook Marketplace RRC (Recommendation Ranking System)

> Facebook Marketplace Search Ranking is a projrct, in which we will develop and train a multimodal model that accepts images 
and text. The output of the model generates embedding vectors that are helpful to make recommendations to buyers using their 
demographic information. 

## Data Cleanning

> This project has two types of tabular and image datasets.

- Cleaning tabular dataset:
    - Product information are stored in 'Products.csv' file. The produucts details includs 'index', 'id', 'product_name', 'category', 'product_description',
       'price', 'location'. 
        - id: is a unique product id
        - product_name: stores three informaton, product name, location and, Gumtree (which is a neme of trading website in the UK). The location has a sparate
        column and the Gumtree does not add any value to the product name, there for this variable has been ruduced to have only the product name.
        - category: mainly stores two info, catagory of the products and extra details regarding the product cotegory. This variable also has been defided to 
        two sections.
        - product_description: includs the description for each product.
        - price: which has around 285 null values. However since we do not need the price value for our project we keep them for our analysis for now
        - location: stores the location of products

- Claining image data sets:
    - We have 12668 images for Facebook Marketing RRS project which all come with different sizes. Around 90.6% of the images have pixel size 
    of greater than 512, ~9.2% have pixel ranges between 256 and 512 and, only ~0.2% of the images have less than 256 pixels.

    - In order to clean our image datset we need to first resize the images with the final size of 64 x 64 pixels to reduce the computing time and have a unique shape. The following function
    will be used for the reszing them:
    ```python
    def resize_image(self, final_size, im):
        
        '''
        Resize image to a specific size
        '''
        size = im.size
        ratio = float(final_size) / max(size)
        new_image_size = tuple([int(x*ratio) for x in size])
        im = im.resize(new_image_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (final_size, final_size))
        new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
        return new_im
    ```

## predict the price of the product

> We will create a simple regression model to predict the price of the products base on a tabular dataset.

- We use the features from our cleaned dataset suuch as product name, product description, and location to predict the price of the product.

- For this task it's necessary to apply TF-IDF (Term Frequency Inverse Document Frequency) to the text data to assign a weight to each word in the text.

- First, we build a simple linear regression model to predict the price of the products.
```python
def lr_clf(self):

    '''
    Builds a simple Linear Regression model to predic the price
    '''
    X, y = self.load_data(True)

    vect = CountVectorizer(ngram_range=(1,3)) # vectorize the text
    train = sp.hstack(X.apply(lambda col: vect.fit_transform(col))) # joins all columns after vectorizing the text

    X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = self.test_to_train_ratio, random_state = self.seed)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    print('Score: ', clf.score(X_test, y_test))
```

As shown above, after loading data, the code will vectorize the text information and joins all text columns after vectorizing. Then, the LinearRegression method 
from sk-learn is called to to predict the price of each product. The score of this model was around 42%

- In order to improve the accuracy one can create a Pipeline to perform the Linear Regression model.
```python
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
```
Creating a pipline as above will increase the accuracy to 49%.

## Predicts the category of each product

> We will create a simple classification model with the image dataset we cleaned previously to predict the category of each product.

- To predict the category of each product from its image we need to do the following steps:
    - clean images and resize them to 64 x 64 pixels
    - change the color images to gray images to simplfy image learning.
    ```code
    img = np.mean(im, axis=2) # convert color image to gray
    ```
    - convert images to numpy arrays
    ```code
    im_array = asarray(img)
    ```
    - convert category information to numerical values
    ```code
    y = (pd.get_dummies(shuffled['category_edited'])).values
    ```
    We have sguffled the dataset to not bais the learning.
    - build a SVC model to predict the category of the products after scaling the image arrays
    ```python
    def svc_model(self):

        '''
        Scales the image arrays and, builds a SVC model to predic the category of the products
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
    ```
    This model only produces an accuracy of ~13% which will be drastically increased with the usage of a CNN model.

    