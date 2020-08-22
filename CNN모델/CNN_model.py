#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


tf.__version__


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical


# In[4]:


get_ipython().system('pip install pillow')
get_ipython().system('pip install sklearn')


# In[5]:


import PIL
from sklearn.model_selection import train_test_split
import random
import os


# In[6]:


print(os.listdir("D:/workspace/AOI/청북공장 이미지"))


# ## define Constants

# In[7]:


FAST_RUN = False
IMAGE_WIDTH = 236
IMAGE_HEIGHT = 236
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# ## prepare training data

# In[8]:


filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/train")


# In[9]:


categories = []


# In[10]:


filenames


# In[11]:


for filename in filenames:
    category = filename.split(' ')[0]
    if category == 'OK':
        categories.append(1)
    else:
        categories.append(0)


# In[12]:


df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})


# In[13]:


df


# In[14]:


df['category'].value_counts().plot.bar()


# ## See sample image

# In[15]:


sample = random.choice(filenames)
sample


# In[16]:


image = load_img("D:/workspace/AOI/청북공장 이미지/train/"+sample)


# In[17]:


plt.imshow(image)


# ## Build Model

# In[18]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization


# In[19]:


from tensorflow.keras.utils import plot_model


# In[20]:


model = Sequential()


# In[21]:


# model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
# filters : 필터(커널)의 개수,
# kernal_size : 필터의 크기
# activation : 활성화 함수
# input_shape : 첫 레이어에 인풋으로 들어오는 크기

# MaxPooling2D
# pool_size : 축소시킬 필터의 크기
# strides : 필터의 이동 간격, 기본값으로 pool_size를 가짐

# Dropout : rate 비율 만큼 drop 시킴
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # 1차원으로 변환하는 layer
model.add(Dense(512, activation='relu')) # unit: 완전연결 layer로 아웃풋 개수
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary() #모델의 layer와 속성 확인


# ## Callbacks

# In[22]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Early Stop

# To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased

# In[24]:


earlystop = EarlyStopping(patience=10)


# Learning Rate Reducation

# We will reduce the learning rate when then accuracy not increase for 2 steps

# In[25]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[26]:


callbacks = [earlystop, learning_rate_reduction]


# ## Prepare Data

# In[27]:


df


# In[28]:


df["category"] = df["category"].replace({0: 'NG', 1: 'OK'}) 


# In[29]:


df


# In[30]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[31]:


train_df['category'].value_counts().plot.bar()


# In[32]:


validate_df['category'].value_counts().plot.bar()


# In[33]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# ## Training Generator

# In[34]:


batch_size


# In[35]:


IMAGE_SIZE


# In[36]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[37]:


train_datagen


# ## Validation Generator

# In[38]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# ### See how our generator work

# In[39]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "D:/workspace/AOI/청북공장 이미지/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[40]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# # Fit Model

# #### epochs : 입력 데이터 학습 횟수
# #### batch_size : 학습할 때 사용되는 데이터 수
# #### verbose : 학습 중 출력되는 로그 수준(0,1,2)

# In[41]:


epochs=5 if FAST_RUN else 5
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# Save Model

# In[42]:


model.save('model.h5')
model.save_weights("model.h5")


# ## Virtualize Training

# In[43]:


history.history


# In[44]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# ## preparing Test Data

# In[45]:


test_filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[46]:


test_df.head()


# ## Create Testing Generator

# In[47]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "D:/workspace/AOI/청북공장 이미지/test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# ## Predict

# In[48]:


predict = model.predict_generator(test_generator, 
                                  steps=np.ceil(nb_samples/batch_size))


# For categoral classication the prediction will come with probability of each category. So we will pick the category that have the highest probability with numpy average max

# In[51]:


test_df['category'] = np.argmax(predict, axis = -1)


# In[52]:


test_df


# We will convert the predict category back into our generator classes by using train_generator.class_indices. It is the classes that image generator map while converting data into computer vision
# 
# 

# In[53]:


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)


# From our prepare data part. We map data with {1: 'dog', 0: 'cat'}. Now we will map the result back to dog is 1 and cat is 0

# In[54]:


test_df['category'] = test_df['category'].replace({ 'NG': 0, 'OK': 1 })


# In[55]:


test_df.head()


# ### virtualize result

# In[56]:


test_df['category'].value_counts().plot.bar()


# In[57]:


IMAGE_SIZE


# In[58]:


sample_test = test_df.head(18)
sample_test.head()
#plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("D:/workspace/AOI/청북공장 이미지/test/"+filename, target_size=IMAGE_SIZE)
   # plt.subplot(4, 3, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()


# In[59]:


submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('S:/submission.csv', index=False)


# In[60]:


submission_df['real'] = submission_df.id.str.split(' ').str[0]


# In[61]:


submission_df


# accuracy print

# In[62]:


score = model.evaluate_generator(validation_generator,steps=len(validation_generator))
print('Test score:', score[0])
print('Test accuracy:', score[1])


# ### confusion matrix

# In[63]:


from sklearn.metrics import confusion_matrix


# In[64]:


pd.crosstab(submission_df['real'], submission_df['label'], rownames=['True'], colnames=['Predicted'], margins=True)


# # 모델 불러오기

# In[65]:


from keras.models import load_model


# In[66]:


model.save('model.h5')


# In[67]:


json_string = model.to_json()


# In[68]:


model = load_model('model.h5')


# In[69]:


model_ss = tf.keras.models.load_model('model_ss.h5')


# In[70]:


model


# In[71]:


model_ss


# In[72]:


import matplotlib.pyplot as plt


# In[73]:


import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image


# In[74]:


#tmp_name = "D:/workspace/AOI/청북공장 이미지/train_ver2/NG (8).jpg"
tmp_name = "D:/workspace/AOI/청북공장 이미지/OK_all/01000.jpg"

test_image = image.load_img(tmp_name, target_size = (236, 236))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


# In[75]:


test_gen = ImageDataGenerator(rescale=1./255)


# In[76]:


test_gen


# In[77]:


#filename : 데이터 들어오는 형식 보고 파일명=위치가 되도록 수정

b_filenames = os.listdir("D:/workspace/AOI/청북공장 이미지/barcode")
b_df = pd.DataFrame({
    'filename': b_filenames
})
nb_samples = b_df.shape[0]


# In[78]:


nb_samples


# In[79]:


test_generator = test_gen.flow_from_dataframe(
    b_df, 
    "D:/workspace/AOI/청북공장 이미지/barcode", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[80]:


test_generator


# In[81]:


test_generator


# In[82]:


tmp_predict = model_ss.predict(test_generator, 
                                          steps=np.ceil(nb_samples/batch_size))


# In[83]:


tmp_predict


# In[84]:


tmp_predict[:1,]


# In[85]:


plt.imshow(load_img(tmp_name))


# In[86]:


tmp = model.predict(test_image)
print(tmp)


# In[87]:


if np.argmax(tmp) > 0:
    print("OK")
else:
    print("NG")


# In[88]:


rs = np.argmax(tmp)
print(rs)


# In[89]:


tmp


# # model graph

# In[90]:


import pydotplus


# In[91]:


pydotplus.find_graphviz()


# In[92]:


from IPython.display import SVG
from tensorflow.python.keras.utils.vis_utils import model_to_dot

get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


import matplotlib.pyplot as plt
get_ipython().system('pip install graphviz')


# In[95]:


SVG(model_to_dot(model,show_shapes=True).create(prog='dot',format='svg'))


# In[ ]:




