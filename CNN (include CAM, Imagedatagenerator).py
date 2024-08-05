#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


tf.reset_default_graph()


# In[ ]:


pine_fir = "./your image file path/"
categories = ["Categories of things you want to categorize"]

nb_classes = len(categories)

image_w = 64
image_h = 64
pixels = image_w * image_h # Take an image size of 64 * 64 and represent it as RGB(3)
x = [] # input data
y = [] # input labeling


# In[ ]:


X_train, X_test, y_train, y_test = np.load("./5obj_1.npy", allow_pickle=True)

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256


# In[ ]:


X_train.shape # The last one corresponds to color, which is RGB(3). 1 means black and white


# # create CNN model

# In[ ]:


model = Sequential() ## this model is very general
## When using this CNN, modify the internals to make it work for you

model.add(Convolution2D(64, (3,3), padding='same', input_shape = X_train.shape[1:]))
model.add(Activation('relu'))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(128, (3,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.run_eagerly = True
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


batch_size = [write you want batch size]
epochs = [write you want epochs]


# In[ ]:


sess = tf.Session()
graph = tf.get_default_graph()


# In[ ]:


history = model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs)


# In[ ]:


plt.figure(figsize=(6,4))
plt.title('acc')
plt.plot(history.history['acc'], 'b', label='accuracy')


# In[ ]:


plt.figure(figsize=(6,4))
plt.title('loss')
plt.plot(history.history['loss'], 'r')


# In[ ]:


score = model.evaluate(X_test, y_test)
print('loss=', score[0])
print('accuracy=', score[1])


# # Channel visualization

# In[ ]:


img_path = './For example, enter the address of the image you want to print'
from keras.preprocessing import image

img = tf.keras.utils.load_img(img_path, target_size=(64, 64))
img_tensor = tf.keras.utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor , axis=0)
img_tensor /= 255.

print(img_tensor.shape)


# In[ ]:


plt.imshow(img_tensor[0])
plt.show()


# In[ ]:


model.layers[:8]


# In[ ]:


from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)


# In[ ]:


activations = activation_model.predict(img_tensor)


# In[ ]:


first_layer_activation = activations[0]
print(first_layer_activation.shape)


# In[ ]:


plt.matshow(first_layer_activation[0, :, :, 62], cmap='viridis')
plt.show()


# In[ ]:


plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
plt.show()


# In[ ]:


layer_activation[0,
                                             :, :,
                                             col * 16 + col]


# In[ ]:


# Use the name of the layer as the graph title
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Draw an feature map
for layer_name, layer_activation in zip(layer_names, activations):
    # Number of features in the feature map
    n_features = layer_activation.shape[-1]

    # feature map size (1, size, size, n_features)
    size = layer_activation.shape[1]

    # Find the grid size for the activation channel
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # Fill each activation into one large grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Process attributes for graphical representation
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # print greed
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()


# # CNN filter visualization

# In[ ]:


from keras import backend as K


# In[ ]:


layer_name = 'conv2d_2'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:, filter_index])


# In[ ]:


# Extract the first tensor from the tensor list returned by the gradients function
grads = K.gradients(loss, model.input)[0]


# In[ ]:


grads /= (K.sqrt(K.mean(K.square(grads))) * 1e-5)


# In[ ]:


iterate = K.function([model.input], [loss, grads])

loss_value, grads_value = iterate([np.zeros((1, 64, 64, 3))])


# In[ ]:


input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# The size of the gradient to update
step = 1.
for i in range(40):   # Run the gradient ascent method 40 times
    # Calculate loss and gradient
    loss_value, grads_value = iterate([input_img_data])
    # Modify the input image in a way that maximizes loss
    input_img_data += grads_value * step


# In[ ]:


def deprocess_image(x):
    # Normalize the tensor to have a mean of 0 and a standard deviation of 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # Clipping to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # Convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# In[ ]:


def generate_pattern(layer_name, filter_index, size=150):
    # Define a loss function to maximize activation for a given layer and filter
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Calculate the gradient of an input image for a loss
    grads = K.gradients(loss, model.input)[0]

    # Gradient Normalization
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # Returns the loss and gradient for an input image
    iterate = K.function([model.input], [loss, grads])
    
    # Start with a noisy gray image
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run 40 steps of the gradient ascent method
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)


# In[ ]:


plt.imshow(generate_pattern('conv2d_2', 0))
plt.show()


# In[ ]:


for layer_name in ['conv2d', 'conv2d_1', 'conv2d_2']:
    size = 64
    margin = 5

    # An empty (black) image to hold the result
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3), dtype='uint8')

    for i in range(8):  # Iterate over the rows in the results grid
        for j in range(8):  # Iterate over the columns in the results grid
            # Generate a pattern for the i + (j * 8)th filter in layer_name
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Save at position (i, j) in the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Draw the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()


# # grad-CAM(Class Activation Map)

# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions


# In[ ]:


cam_img_path = './your image file path'

cam_img = tf.keras.utils.load_img(cam_img_path, target_size=(224, 224)) # Returns as a PIL object of size 224 x 224
x = tf.keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# In[ ]:


NUM_CLASS = 3


# In[ ]:


korean_pine_output = model.output[: , 1]
last_conv_layer = model.get_layer('conv2d_2')


# In[ ]:


grads = K.gradients(korean_pine_output, last_conv_layer.output)[0]


# In[ ]:


pooled_grads = K.mean(grads, axis=(0,1,2))


# In[ ]:


iterate = K.function([model.input],
                        [pooled_grads, last_conv_layer.output[0]])


# In[ ]:


pooled_grads_value, conv_layer_output_value = iterate([x])


# In[ ]:


for i in range(NUM_CLASS):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]


# In[ ]:


heatmap = np.mean(conv_layer_output_value, axis = -1)
gradCAM = np.maximum(heatmap, 0)


# In[ ]:


gradCAM /= np.max(gradCAM)
plt.matshow(heatmap)


# # Overlaying a CAM Source Image

# In[ ]:


import cv2


# In[ ]:


cam_img


# In[ ]:


origin_img = cv2.imread(cam_img_path)
heatmap = cv2.resize( heatmap, (224, 224))
heatmap = np.uint8( 255 * heatmap )
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + cam_img

cv2.imwrite('./The address of the image you want to return', superimposed_img)


# # Initialize the model

# In[ ]:


K.clear_session()

