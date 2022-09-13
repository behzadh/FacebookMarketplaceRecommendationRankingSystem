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

    - In order to clean our image datset we need to first resize the images with the final size of 512 x 512 pixels. The following function
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
