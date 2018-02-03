from glob import glob
import os
from feat.bag_of_visual_words import BagOfVisualWords

class RectagleImages(object):

    def __init__(self, config = None):
        self.config = config

    def read_image_paths(self):

        IMG_DIR = '/media/d1/data/img-rectangles/imgs'

        # Fetch the paths to images.
        zero_imgs = []
        for img_0 in glob(os.path.join(IMG_DIR,'0','*.jpg')):
            zero_imgs.append(img_0)

        one_imgs = []
        for img_1 in glob(os.path.join(IMG_DIR, '1', '*.jpg')):
            one_imgs.append(img_1)


        return zero_imgs, one_imgs




if __name__ == '__main__':

    obj = RectagleImages()

    zero_imgs, one_imgs = obj.read_image_paths()
    print(zero_imgs, one_imgs)

    bovw = BagOfVisualWords()
    bovw.train_model(images=[zero_imgs,one_imgs])



