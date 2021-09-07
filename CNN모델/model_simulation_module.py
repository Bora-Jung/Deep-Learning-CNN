
import numpy as np
import pandas as pd
import os
import MySQLdb
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img

# 모델 호출
model_ss = tf.keras.models.load_model('model_ss.h5')

# 폴더 지정
n = 1 #몇번째 폴더인지?
path = "D:/workspace/AOI/청북공장 이미지/simulation"
load = os.listdir(path)
file_list = os.listdir(path + '/' + load[n])

# 폴더 내 파일 df에 입력
df = pd.DataFrame({
    'file_name': file_list})
df['COMPONENT'] = df.file_name.str.split('_').str[1]
df['COMPONENT'] = df.file_name.str.split('.').str[0]
df['BARCODE'] = load[n]

# 예측을 위한 값 설정
FAST_RUN = False
IMAGE_WIDTH = 236
IMAGE_HEIGHT = 236
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
batch_size = 3

# generator 설정
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    df,
    path + '/' + load[n],
    x_col = 'file_name',
    y_col = None,
    class_mode = None,
    target_size = IMAGE_SIZE,
    batch_size = batch_size,
    shuffle = False)

# 예측
tmp_predict = model_ss.predict(test_generator)

# 예측 결과 정리
df['PRED_NG'] = np.round(tmp_predict[:,0], 4)
df['PRED_OK'] = np.round(tmp_predict[:,1], 4)

conditions = [
    (df['PRED_OK'] > df['PRED_NG']),
    (df['PRED_OK'] < df['PRED_NG'])]
choices = ['OK', 'NG']
df['PRED'] = np.select(conditions, choices, default='null')

# df 정리
del df['file_name']
df['PATH'] = 'path' # 이미지 경로 지정
df = df[['BARCODE', 'COMPONENT', 'PATH', 'PRED_OK', 'PRED_NG', 'PRED']]


# df to db
conn = MySQLdb.connect(db='MONITERING', user = 'ybigLF', passwd = '!Yura@@', port=3306, host='172.21.0.67')
c = conn.cursor()

df1 = df.values.tolist()
c.executemany('INSERT INTO AOI_PRED(BARCODE, COMPONENT, PATH, PRED_OK, PRED_NG, PRED)VALUES (%s,%s,%s,%s,%s,%s)',df1)

conn.commit()

from keras.models import load_model
model_ss = tf.keras.models.load_model('model_ss.h5')
model_ss


# 폴더 지정
path = "D:/workspace/AOI/청북공장 이미지/simulation"
load = os.listdir(path)
file_list = os.listdir(path + '/' + load[n])

df = pd.DataFrame({
    'file_name': file_list})

df['COMPONENT'] = df.file_name.str.split('_').str[1]
df['COMPONENT'] = df.file_name.str.split('.').str[0]

df['BARCODE'] = load[n]

FAST_RUN = False
IMAGE_WIDTH = 236
IMAGE_HEIGHT = 236
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
batch_size = 3

test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    df,
    path + '/' + load[n],
    x_col = 'file_name',
    y_col = None,
    class_mode = None,
    target_size = IMAGE_SIZE,
    batch_size = batch_size,
    shuffle = False
)

tmp_predict = model_ss.predict(test_generator)
tmp_predict

df['PRED_NG'] = np.round(tmp_predict[:,0], 4)
df['PRED_OK'] = np.round(tmp_predict[:,1], 4)

conditions = [
    (df['PRED_OK'] > df['PRED_NG']),
    (df['PRED_OK'] < df['PRED_NG'])]
choices = ['OK', 'NG']
df['PRED'] = np.select(conditions, choices, default='null')

del df['file_name']


df['PATH'] = 'tmp'
df['DATETIME'] = df.BARCODE.str.split('_').str[7]
df = df[['BARCODE', 'COMPONENT', 'PATH', 'PRED_OK', 'PRED_NG', 'PRED', 'DATETIME']]
df


# export DB

get_ipython().system('pip install PyMySQL')
import pymysql
from sqlalchemy import create_engine

get_ipython().system('pip install PyMySQL')
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

conn = MySQLdb.connect(db='MONITERING', user = 'ybigLF', passwd = '!Yura@@', port=3306, host='172.21.0.67')
c = conn.cursor()

df1 = df.values.tolist()
c.executemany('INSERT INTO AOI_PRED(BARCODE, COMPONENT, PATH, PRED_OK, PRED_NG, PRED, DATETIME)VALUES (%s,%s,%s,%s,%s,%s, %s)', df1)

conn.commit()

user = 'ybigLF'
passw = '!Yura@@'
Host =  '172.21.0.67'  # either localhost or ip e.g. '172.17.0.2' or hostname address 
port = 3306 
database = 'MONITERING'

engine = create_engine('mysql+pymysql://' + user + ':' + passw + '@' + Host + ':' + str(port) + '/' + database , echo=False)
conn = engine.connect()
df.to_sql(name = 'AOI_PRED', con = engine, if_exists='append', index=False)
