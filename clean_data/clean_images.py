'''
Cleaning an image dataset.
@author:    Behzad 
@date:      10 September 2022
'''
from numpy import asarray
from PIL import Image
import pandas as pd
import os
from clean_tabular_data import CleanData

class CleanImage(CleanData):
    
    '''
    This class will clean text data and has the following methods:

    resize_image(self, final_size, im)
    clean_image_data(self)
    '''
    def __init__(self) -> None:
        self.dir = '/Users/behzad/AiCore/fb_raw_data/'
        self.final_size = 64
        
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

    def clean_image_data(self):

        '''
        Cleans image by resizing them to 512 x 512 pixel
        '''
        path = self.dir + "images/"
        dirs = os.listdir(path)
        for n, item in enumerate(dirs, 1):
            im = Image.open(path + item)
            # width, height = im.size
            # print(width, height)
            new_im = self.resize_image(self.final_size, im)
            new_im.save(self.dir + f'cleaned_images/{item[:-4]}_resized.jpg')

    def image_to_array(self):

        '''
        Converts image to a numpy array and save the in a pickle file
        '''
        path = self.dir + "cleaned_images/"
        df = self.clean_text_data()
        profuct_id_list = df['id'].to_list()
        img_df = pd.DataFrame()
        for id in profuct_id_list:
            im = Image.open(path + id + '_resized.jpg')
            #img = np.mean(im, axis=2) # convert color image to gray
            im_array = asarray(im)
            img_df = img_df.append({"image": [im_array], 'id': id}, ignore_index=True)
        df_final = pd.merge(df, img_df)
        df_final.to_pickle('./ml_models/products_w_imgs.pkl')

if __name__ == '__main__':
    cln_img = CleanImage()
    #cln_img.clean_image_data()
    cln_img.image_to_array()