# 본 프로젝트는 파이썬 기반 웹 서비스이다
# keras롤 학습된 MNIST 이미지 분류 모델을 이식하여 
# 웹기반으로 이미지를 판독해주는 서비스이다

# 주요 버전
tensorflow 1.15.2
(keras는 tensorflow.keras로 사용 빌트인)

# 구조
/
L run.py          : 프로그램 시작점, 엔트리 포인트
                    모든 경로는 여기서부터 시작함
L h5              : 사전에 오프라인으로 학습된 모델이 위치하는 폴더
    L mnist.h5    : 최초로 학습한 모델 파일
L model           : 패키지명
    L __init__.py : 모델을 이용하여 예측을 수행하는 기능
                    데이터 전처리 기능 제공(예측시 데이터형태를 맞춰주는 기능)
L templates       : 웹서비스 쪽에서 html을 랜더링 할때 참조되는 위치
                    html이 위치하는곳
    L index.html  : 파일 업로드 화면
L requirements.txt: 본 서비스에 사용된 모든 패키지의 버전기술
                    차후 $ pip install -r requirements.txt로 설치


# 구동
- 기본적으로 다른 가상환경에 진입하였으므로, 가상환경 종료 처리
(base)$ conda deactivate

- 우리 프로젝트의 가상환경으로 진입
$ conda activate dl_service

- 설치 패키지 목록에서 flask, tensorflow 확인
(dl_service)$ conda list

- 서버가동
(dl_service)$ python run.py

- 접속
http://127.0.0.1:5000 or http://localhost:5000
or http://서버가구동중인 서버공인IP or 내부IP(로컬에서만):5000
- 실제 서비스로 올릴때는 방화벽 오픈(인바운드)