#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pine_dir = "./file address/"
categories = ["The species you want to categorize"]
nb_classes = len(categories)

image_w = 64 ## Enter the desired image size
image_h = 64
pixels = image_w * image_h * 3

x = []
y = [] # For labeling 

import glob # Read a file into a glob
import numpy as np
from PIL import Image # Image fetching libraries

for idx, cat in enumerate(categories):
    #print(idx)
    #print(cat)
    label = [0 for i in range(nb_classes)]
    #print(label)
    label[idx] = 1
    #print(label)
    image_dir = pine_dir + "" + cat
    print(image_dir)
    files = glob.glob(image_dir + "/*.jpg") # 해Read data from those files.
    #print(files)
    for i, f in enumerate(files):
        #print(i)
        #print(f)
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        #print(data)
        x.append(data)
        y.append(label)
        #if i% 100 == 0:
        #    print(i,"\n", data)

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) ## ignore warning

x = np.array(x)
y = np.array(y)


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)
np.save("./save file name", xy)

print("ok....", len(y))

