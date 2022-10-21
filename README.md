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

## Creating Vision Model

> In this section we will train a PyTorch CNN model to classify the category of products based on their images.

- PyTorch Dataset:
    - To create a PyTorch dataset we can simply use the 'torch.utils.data.Dataset' module. The idea is to create a PyTorch dataset 
    which includs an image tensor as features and the category of products as target.

- Build Convolutional Neural Network:
    - A CNN model with two hidden layers designed for the image classification. 
    
    ```python
    class CNN(nn.Module):

        '''
        A two layers cnn model used for image classification 
        '''

        def __init__(self, num_classes=3) -> None:

            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(3, 12, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Conv2d(12,24, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(2,2),
                nn.Dropout2d(p=0.2),
            
                nn.Flatten(),
                nn.Linear(6144, num_classes), #28800
            )

        def forward(self, x):
            return self.layers(x)
    ```
    This model has reached an accurancy of 23-24% which is an excepted accurancy based on our dataset. It's around twice better 
    than the previous model.

- Transfer Learning:
    - Can we even do better? Can we use more pictures to improve the accurancy? Where can find the new pictures? The answer
    of these questions is 'yes' we can use pre-trained models that's been trained by others off the shelf. In this project, 
    we use transfer learning to fine tune RESNET-50 to model a CNN that can classify the images using the dataset created 
    previousely.
    We just need to replace the final linear layer of the model with another linear layer whose output size is the same as 
    the number of our categories.
    ```python
    class ResNet50(nn.Module):

        '''
        ResNet50 class is a pre-trained model tuned for our image classification 
        '''

        def __init__(self, num_classes=3) -> None:

            super().__init__()
            self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
            out_features = self.resnet50.fc.out_features #It'd be 2048 for resnet50
            self.linear = nn.Linear(out_features, num_classes).to(device)
            self.layers = nn.Sequential(self.resnet50, self.linear).to(device)

        def forward(self, x):
            return self.layers(x)
    ```
    Using pre-trained resnet50 model increased the accuracy to around 35%.

    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/tensorboard.png" width="600">
    <br />
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/loss.png" width="298">
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/accuracy.png" width="300">

