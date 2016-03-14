
# coding: utf-8

# In[104]:

from wand.image import Image
from wand.color import Color
from wand.display import display
import numpy as np
path = 'image_da.tif'


# In[167]:

def resize_shortest(img):
    x,y = float(img.size[0]), float(img.size[1])
    if x < y:
        xx = 224.0
        yy = y/x*xx
    else:
        yy = 224.0
        xx = (x/y)*yy
    xx,yy = int(xx),int(yy)
    img.resize(xx,yy)


# In[168]:

def random_crop(img):
    x,y = float(img.size[0]), float(img.size[1])
    delta_x = x-224
    delta_y = y-224
    xx = int(np.random.uniform(0, delta_x))
    yy = int(np.random.uniform(0, delta_y))
    img.crop(xx, yy, xx+224, yy+224)


# In[173]:

def rotate(img, max_deg=360):
    deg = int(np.random.uniform(0, max_deg))
    img.rotate(deg, background=Color('rgb(132,132,132)'))
    l,L = img.size
    img.crop(width=int(0.6*l), height=int(0.6*L), gravity='center')


# In[174]:




# In[163]:

[name, ext] = path.split('.')
print name, ext


# In[188]:

from os import listdir
from os.path import isfile, join
import os

d = os.getcwd()
path = listdir('/mnt/mass_2x_padding_dataset_train')[0]


# In[195]:

source_to_augmented = {'/mnt/mass_2x_padding_dataset_train': '/mnt/2x_augmented_train',
                       '/mnt/mass_2x_padding_dataset_test': '/mnt/2x_augmented_test',
                       '/mnt/mass_2x_padding_dataset_val': '/mnt/2x_augmented_val'}


# In[198]:

for directory in source_to_augmented:
    files = listdir(directory)
    N = len(files)
    for i, filepath in enumerate(files):
        img = Image(filename=directory+'/'+filepath)
        [name, ext] = filepath.split('.')
        for t in range(5):
            z = img.clone()
            if t != 0:
                rotate(z)
            resize_shortest(z)
            for tt in range(5):
                new_path = '%s/%s_%i_%i.%s' % (source_to_augmented[directory], name, t, tt, ext)
                n = z.clone()
                random_crop(n)
                n.save(filename=new_path)
        print 'Done %i/%i' % (i,N)


# In[ ]:



