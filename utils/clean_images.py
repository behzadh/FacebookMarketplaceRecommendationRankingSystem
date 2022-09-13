'''
Cleaning an image dataset.
@author:    Behzad 
@date:      10 September 2022
'''
from hashlib import new
from PIL import Image
import os

class CleanImage:
    
    '''
    This class will clean text data and has the following methods:

    resize_image(self, final_size, im)
    clean_image_data(self)
    '''
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
        path = "../../fb_raw_data/images/"
        dirs = os.listdir(path)
        final_size = 512
        for n, item in enumerate(dirs, 1):
            im = Image.open(path + item)
            # width, height = im.size
            # print(width, height)
            new_im = self.resize_image(final_size, im)
            new_im.save(f'../../fb_raw_data/cleaned_images/{item[:-4]}_resized.jpg')

if __name__ == '__main__':
    cln_img = CleanImage()
    cln_img.clean_image_data()