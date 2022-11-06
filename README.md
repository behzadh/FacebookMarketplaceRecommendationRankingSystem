# Facebook Marketplace RRC (Recommendation Ranking System)

> Facebook Marketplace Search Ranking is a project, in which we will develop and train a multimodal model that accepts images 
and text. The output of the model generates embedding vectors that are helpful to make recommendations to buyers using their 
demographic information. 

## Data Cleanning

> This project has two types of tabular and image datasets.

- Cleaning tabular dataset:
    - Product information is stored in the 'Products.csv' file. The products details include 'index', 'id', 'product_name', 'category', 'product_description',
       'price', 'location'. 
        - id: is a unique product id
        - product_name: stores three pieces of information, product name, location and, Gumtree (which is the name of a trading website in the UK). The location has a separate
        column and word Gumtree do not add any value to the product name, there for this variable has been reduced to have only the product name.
        - category: mainly stores two pieces of info, the category of the products and extra details regarding the product category. This variable also has been defined to 
        two sections.
        - product_description: include the description for each product.
        - price: which has around 285 null values. However, since we do not need the price value for our project we keep them for our analysis for now
        - location: stores the location of products

- Cleaning image data sets:
    - We have 12668 images for Facebook Marketing RRS project which all come in different sizes. Around 90.6% of the images have a pixel size 
    greater than 512, ~9.2% have pixel ranges between 256 and 512 and, only ~0.2% of the images have less than 256 pixels.

    - To clean our image datset we need to first resize the images with the final size of 64 x 64 pixels to reduce the computing time and have 
    a unique shape. The following function will be used for the resizing them:
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

## Price Prediction

> We will create a simple regression model to predict the price of the products base on a tabular dataset.

- We use the features from our cleaned dataset such as product name, product description, and location, to predict the price of each product.

- For this task, it's necessary to apply TF-IDF (Term Frequency Inverse Document Frequency) to the text data to assign a weight to each word within the text.

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
from sk-learn is called, to predict the price of each product. The score of this model was around 42%

- To improve the accuracy one can create a Pipeline to perform the Linear Regression model.
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
Creating a pipeline as above will increase the accuracy to 49%.

## Prediction of Different Categories

> We will create a simple classification model with the image dataset we cleaned previously to predict the category of each product.

- To predict the category of each product from its image we need to do the following steps:
    - clean images and resize them to 64 x 64 pixels
    - change the colour images to grey images to simplify image learning.
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
    We have shuffled the dataset to not bias the learning.
    - build an SVC model to predict the category of the products after scaling the image arrays
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

## Vision Model

> In this section, we will train a PyTorch CNN model to classify the category of products based on their images.

- PyTorch Dataset:
    - To create a PyTorch dataset we can simply use the 'torch.utils.data.Dataset' module. The idea is to create a PyTorch dataset 
    which includes an image tensor as a feature and the category of products as a target.

- Build Convolutional Neural Network:
    - A CNN model with two hidden layers designed for image classification. 
    
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
    This model has reached an accuracy of 23-24%, which is an excepted accuracy based on our dataset. It's around twice better 
    then the previous model.

- Transfer Learning:
    - Can we even do better? Can we use more pictures to improve the accuracy? Where can we find new pictures? The answer
   is 'yes' we can use pre-trained models that have been trained by others off the shelf. In this project, 
    we use transfer learning to fine-tune RESNET-50 to model a CNN that can classify the images using the same dataset created 
    previously.
    We only need to replace the final linear layer of the model with another linear layer whose output size is the same as 
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
    Using the pre-trained resnet50 model increased the accuracy to around 35% for the validation and the accuracy for the training 
    sample reached 65% accepting an overfit for our model, based on the available data for this project.

    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/image_tensorboard.png" width="600">
    <br />
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/image_loss.png" width="298">
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/image_accuracy.png" width="300">

## Text Model

> In this section, we will train a PyTorch CNN model to classify the category of products based on their text description.

- PyTorch Dataset:
    - Same as the Image Dataset, we use the 'torch.utils.data.Dataset' module. This PyTorch dataset 
    will include an embedding tensor as a feature and the category of products as the target.

    - Before embedding the raw text data is cleaned to drop unnecessary information for the training model.

- Build Convolutional Neural Network:
    - In this stage, our dataset is very similar to the image dataset. We just need to use a 1d (Conv1d) CNN model 
    instead of the 2d (Conv2d) model we used for the image classification. 
    
    ```python
    class CNN(nn.Module):

        '''
        A cnn model used for text classification 
        '''

        def __init__(self, num_classes=3) -> None:

            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv1d(768, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.layers(x)
    ```
    This model has reached an accuracy of 74%, which is a very good result for this dataset. the accuracy of the training 
    sample reached 84%, and no overfitting occurred within the five epochs.

    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_tensorboard.png" width="600">
    <br />
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_loss.png" width="298">
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_accuracy.png" width="300">

## Combined Model

> Once we have the Vision and Text models, we can combine them by concating their last layers.

- PyTorch Dataset:
    - Same as the Image and Text Dataset we create a dataset to include both pieces of information. This PyTorch dataset 
    will include embedding and image tensors as features and the category of products as the target.


- Combine the two models:
    - This will be a model that concatenates the layers of the image and text models.
    
    ```python
    class CombinedModel(nn.Module):

        '''
        A combined model used for text and image classification together
        '''

        def __init__(self, num_classes=3) -> None:

            super(CombinedModel, self).__init__()
            resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
            out_features = resnet50.fc.out_features
            self.image_classifier = nn.Sequential(resnet50, nn.Linear(out_features, 128)).to(device)
            self.text_classifier = TextClassifier()
            self.main = nn.Sequential(nn.Linear(192, num_classes))

        def forward(self, image_features, text_features):
            image_features = self.image_classifier(image_features)
            text_features = self.text_classifier(text_features)
            combined_features = torch.cat((image_features, text_features), 1)
            combined_features = self.main(combined_features)
            return combined_features
    ```
    The combined model, which uses both test and image information has reached an accuracy of 78%, which so far is one of the 
    best results at the AiCore. The accuracy for the training sample reached 97% within the eight epochs. 

    ```code
    Epoch: 1
    Training set: Average loss: 1.714934, Accuracy: 3751/8823 (43%)
    Validation set: Average loss: 1.301012, Accuracy: 2137/3781 (57%)

    Epoch: 2
    Training set: Average loss: 1.068653, Accuracy: 5727/8823 (65%)
    Validation set: Average loss: 1.129872, Accuracy: 2394/3781 (63%)

    Epoch: 3
    Training set: Average loss: 0.765921, Accuracy: 6605/8823 (75%)
    Validation set: Average loss: 1.019578, Accuracy: 2563/3781 (68%)

    Epoch: 4
    Training set: Average loss: 0.546157, Accuracy: 7284/8823 (83%)
    Validation set: Average loss: 1.095542, Accuracy: 2612/3781 (69%)

    Epoch: 5
    Training set: Average loss: 0.411921, Accuracy: 7667/8823 (87%)
    Validation set: Average loss: 0.963552, Accuracy: 2722/3781 (72%)

    Epoch: 6
    Training set: Average loss: 0.305380, Accuracy: 7955/8823 (90%)
    Validation set: Average loss: 0.922950, Accuracy: 2855/3781 (76%)

    Epoch: 7
    Training set: Average loss: 0.212714, Accuracy: 8213/8823 (93%)
    Validation set: Average loss: 0.974951, Accuracy: 2886/3781 (76%)

    Epoch: 8
    Training set: Average loss: 0.103855, Accuracy: 8544/8823 (97%)
    Validation set: Average loss: 1.008297, Accuracy: 2935/3781 (78%)
    ```

    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_image_tensorboard.png" width="600">
    <br />
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_image_loss.png" width="300">
    <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/text_image_accuracy.png" width="300">

## Deploy Model on AWS Server:

> Once both vision and text models are combined, we can deploy our model on a server. In this section, we will
discuss the FastApi used for deploying our model on an AWS EC2 instance.

- API (Application Programming Interface):
    - Application programming interfaces are software development and innovation by enabling 
    applications to exchange data and functionality.
    - API, enables companies to open up their applications’ data and functionality to external 
    third-party developers and business partners, or to departments within their companies.
    - In other words, APIs sit between an application and the web server, acting as an intermediary 
    layer that processes data transfer between systems.

- FastAPIs:
    - In this project, we used FastAPI to create three endpoints to predict image, text and combined image and text data.
    - FastAPI is a modern, high-performance web framework for building APIs with Python based on standard type hints.
    - We developed our API with all the models and training weights for each model in the api.py file.
    - Then, we containerized our API to deploy it on an EC2 instance.

- Results:
    - Now that our API has been deployed, we can send some requests to all three endpoints with the relevant data and test them to 
    return predictions and probabilities. 

    - Our API looks like the following with three endpoints:
        <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/Api_1.png" width="600">
            <br />
    - We can test it by inserting some data, like a text data as an example:
        <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/Api_2.png" width="600">
            <br />
    - And, it correctly predicts its category as 'Phones, Mobile Phones & Telecoms':
        <img src="https://github.com/behzadh/Facebook_Marketplace_RRS/blob/main/plots/Api_3.png" width="600">
            <br />
