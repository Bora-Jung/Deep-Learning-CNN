# 모델 로드 (서버가 가동하면 1회 로드된다)
import tensorflow as tf
from tensorflow.keras.utils import *
from tensorflow.keras.models import load_model

print( tf.__version__ )
# 모든 경로법은 기준 => 엔트리 포인트 기준(run.py)
#model = load_model('./h5/mnist.h5')

# pgm 파일을 읽어서 예측가능한 형태로 전처리해주는 함수
import numpy as np

def decodePGM( path ):
  with open(path, 'r', encoding='utf-8') as f:
    src = f.read()
  tmp = src.split('\n')[-1].split()
  tmp = list( map( int, tmp) )  
  tmp = np.array( tmp, dtype='float32' )
  tmp = tmp/255
  return tmp.reshape(-1, 28, 28, 1)

# model 패키지 자체를 의미
def predict_mnist( path ):
    # 1. 파일을 읽는다 
    # 2. 전처리를 통해서 예측이 가능한 형태로 데이터를 가공
    data     = decodePGM( path )
    # 3. 모델에 데이터를 주입(입력)하여 예측 수행
    model    = load_model('./h5/mnist.h5')
    pred_num = model.predict_classes( data )
    print( pred_num )
    # 4. 응답 데이터를 구성, 응답
    return { 'label':int(pred_num[0]) }