```python
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
from tensorflow import keras
```


```python
# 통합 코드
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

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

import MySQLdb
conn = MySQLdb.connect(db='MONITERING', user = 'ybigLF', passwd = '!Yura@@', port=3306, host='172.21.0.67')
c = conn.cursor()

df1 = df.values.tolist()
c.executemany('INSERT INTO AOI_PRED(BARCODE, COMPONENT, PATH, PRED_OK, PRED_NG, PRED)\
VALUES (%s,%s,%s,%s,%s,%s)',df1)

conn.commit()
```


```python
from keras.models import load_model
model_ss = tf.keras.models.load_model('model_ss.h5')
model_ss
```




    <tensorflow.python.keras.engine.sequential.Sequential at 0x22487b529c8>




```python
n = 8
```


```python
# 폴더 지정
path = "D:/workspace/AOI/청북공장 이미지/simulation"
load = os.listdir(path)
file_list = os.listdir(path + '/' + load[n])
```


```python
file_list
```




    ['NG (1).jpg',
     'NG (2).jpg',
     'NG (3).jpg',
     'OK (1).jpg',
     'OK (2).jpg',
     'OK (3).jpg',
     'OK (4).jpg',
     'OK (5).jpg']




```python
df = pd.DataFrame({
    'file_name': file_list})

df['COMPONENT'] = df.file_name.str.split('_').str[1]
df['COMPONENT'] = df.file_name.str.split('.').str[0]

df['BARCODE'] = load[n]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_name</th>
      <th>COMPONENT</th>
      <th>BARCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>NG (1)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (2).jpg</td>
      <td>NG (2)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (3).jpg</td>
      <td>NG (3)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OK (1).jpg</td>
      <td>OK (1)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK (2).jpg</td>
      <td>OK (2)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OK (3).jpg</td>
      <td>OK (3)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OK (4).jpg</td>
      <td>OK (4)</td>
      <td>test_folder_name</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OK (5).jpg</td>
      <td>OK (5)</td>
      <td>test_folder_name</td>
    </tr>
  </tbody>
</table>
</div>




```python
FAST_RUN = False
IMAGE_WIDTH = 236
IMAGE_HEIGHT = 236
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
batch_size = 3
```


```python
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

test_generator
```

    Found 8 validated image filenames.
    




    <keras.preprocessing.image.DataFrameIterator at 0x2248826b808>




```python
tmp_predict = model_ss.predict(test_generator)
tmp_predict
```




    array([[9.9833763e-01, 1.6623241e-03],
           [9.8982692e-01, 1.0173124e-02],
           [9.9976403e-01, 2.3597175e-04],
           [1.6467227e-02, 9.8353279e-01],
           [3.9497128e-01, 6.0502869e-01],
           [1.0979590e-03, 9.9890208e-01],
           [7.0295632e-01, 2.9704371e-01],
           [5.3797867e-03, 9.9462020e-01]], dtype=float32)




```python
df['PRED_NG'] = np.round(tmp_predict[:,0], 4)
df['PRED_OK'] = np.round(tmp_predict[:,1], 4)

conditions = [
    (df['PRED_OK'] > df['PRED_NG']),
    (df['PRED_OK'] < df['PRED_NG'])]
choices = ['OK', 'NG']
df['PRED'] = np.select(conditions, choices, default='null')

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file_name</th>
      <th>COMPONENT</th>
      <th>BARCODE</th>
      <th>PRED_NG</th>
      <th>PRED_OK</th>
      <th>PRED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1).jpg</td>
      <td>NG (1)</td>
      <td>test_folder_name</td>
      <td>0.9983</td>
      <td>0.0017</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (2).jpg</td>
      <td>NG (2)</td>
      <td>test_folder_name</td>
      <td>0.9898</td>
      <td>0.0102</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (3).jpg</td>
      <td>NG (3)</td>
      <td>test_folder_name</td>
      <td>0.9998</td>
      <td>0.0002</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OK (1).jpg</td>
      <td>OK (1)</td>
      <td>test_folder_name</td>
      <td>0.0165</td>
      <td>0.9835</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK (2).jpg</td>
      <td>OK (2)</td>
      <td>test_folder_name</td>
      <td>0.3950</td>
      <td>0.6050</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OK (3).jpg</td>
      <td>OK (3)</td>
      <td>test_folder_name</td>
      <td>0.0011</td>
      <td>0.9989</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OK (4).jpg</td>
      <td>OK (4)</td>
      <td>test_folder_name</td>
      <td>0.7030</td>
      <td>0.2970</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OK (5).jpg</td>
      <td>OK (5)</td>
      <td>test_folder_name</td>
      <td>0.0054</td>
      <td>0.9946</td>
      <td>OK</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df['file_name']
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>COMPONENT</th>
      <th>BARCODE</th>
      <th>PRED_NG</th>
      <th>PRED_OK</th>
      <th>PRED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NG (1)</td>
      <td>test_folder_name</td>
      <td>0.9983</td>
      <td>0.0017</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NG (2)</td>
      <td>test_folder_name</td>
      <td>0.9898</td>
      <td>0.0102</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NG (3)</td>
      <td>test_folder_name</td>
      <td>0.9998</td>
      <td>0.0002</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OK (1)</td>
      <td>test_folder_name</td>
      <td>0.0165</td>
      <td>0.9835</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK (2)</td>
      <td>test_folder_name</td>
      <td>0.3950</td>
      <td>0.6050</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OK (3)</td>
      <td>test_folder_name</td>
      <td>0.0011</td>
      <td>0.9989</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OK (4)</td>
      <td>test_folder_name</td>
      <td>0.7030</td>
      <td>0.2970</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OK (5)</td>
      <td>test_folder_name</td>
      <td>0.0054</td>
      <td>0.9946</td>
      <td>OK</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['PATH'] = 'tmp'
```


```python
df['DATETIME'] = df.BARCODE.str.split('_').str[7]
```


```python
df = df[['BARCODE', 'COMPONENT', 'PATH', 'PRED_OK', 'PRED_NG', 'PRED', 'DATETIME']]
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BARCODE</th>
      <th>COMPONENT</th>
      <th>PATH</th>
      <th>PRED_OK</th>
      <th>PRED_NG</th>
      <th>PRED</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>test_folder_name</td>
      <td>NG (1)</td>
      <td>tmp</td>
      <td>0.0017</td>
      <td>0.9983</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test_folder_name</td>
      <td>NG (2)</td>
      <td>tmp</td>
      <td>0.0102</td>
      <td>0.9898</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test_folder_name</td>
      <td>NG (3)</td>
      <td>tmp</td>
      <td>0.0002</td>
      <td>0.9998</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>3</th>
      <td>test_folder_name</td>
      <td>OK (1)</td>
      <td>tmp</td>
      <td>0.9835</td>
      <td>0.0165</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>test_folder_name</td>
      <td>OK (2)</td>
      <td>tmp</td>
      <td>0.6050</td>
      <td>0.3950</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>5</th>
      <td>test_folder_name</td>
      <td>OK (3)</td>
      <td>tmp</td>
      <td>0.9989</td>
      <td>0.0011</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>6</th>
      <td>test_folder_name</td>
      <td>OK (4)</td>
      <td>tmp</td>
      <td>0.2970</td>
      <td>0.7030</td>
      <td>NG</td>
    </tr>
    <tr>
      <th>7</th>
      <td>test_folder_name</td>
      <td>OK (5)</td>
      <td>tmp</td>
      <td>0.9946</td>
      <td>0.0054</td>
      <td>OK</td>
    </tr>
  </tbody>
</table>
</div>



## 예측 결과 into DB


```python
!pip install PyMySQL
import pymysql
from sqlalchemy import create_engine

!pip install PyMySQL
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb
```

    Requirement already satisfied: PyMySQL in c:\users\admin\anaconda3\envs\tf\lib\site-packages (0.9.3)
    


```python
import MySQLdb
conn = MySQLdb.connect(db='MONITERING', user = 'ybigLF', passwd = '!Yura@@', port=3306, host='172.21.0.67')
c = conn.cursor()

df1 = df.values.tolist()
c.executemany('INSERT INTO AOI_PRED(BARCODE, COMPONENT, PATH, PRED_OK, PRED_NG, PRED, DATETIME)\
VALUES (%s,%s,%s,%s,%s,%s, %s)', df1)

conn.commit()
```


```python

```


```python
user = 'ybigLF'
passw = '!Yura@@'
Host =  '172.21.0.67'  # either localhost or ip e.g. '172.17.0.2' or hostname address 
port = 3306 
database = 'MONITERING'

engine = create_engine('mysql+pymysql://' + user + ':' + passw + '@' + Host + ':' + str(port) + '/' + database , echo=False)
conn = engine.connect()
df.to_sql(name = 'AOI_PRED', con = engine, if_exists='append', index=False)
```


```python

```
