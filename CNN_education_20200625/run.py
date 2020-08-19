# run.py
# 파이썬의 웹기술 flask(마이크로에디션). DJango(풀스팩에디션)
# 이중 자유도가 높고, 아주 짧은 코드로도 서비스 구축이 가능한 flask 사용
# flask는 스타일이 nodejs와 유사
# DJango는 스타일이 spring과 유사
# 1. 모듈가져오기
from flask import Flask, render_template, request, jsonify,json 
from model import predict_mnist

# 2. Flask 객체 생성
# __name__: "__main__" or "파일명"
app = Flask(__name__)

# 3. 라우팅
# restful 기능 적용
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET': # 화면폼을 전송한다
        return render_template('index.html')
    else:# pgm 파일을 받아서 -> 전처리 -> 예측 -> 응답 
        # 1. 전송된 데이터를 받아서 파일에 저장
        f = request.files['pgm']
        # 2. 파일 저장
        f.save( f.filename )
        # 3. 파일명(경로)를 보내서, 예측값을 받는다
        res = predict_mnist( f.filename ) 
        print( type(res) )       
        return jsonify( res )

# 4. 서버 가동
if __name__ == '__main__': # 이 코드를 직접 구동했다면
    # 적절하게 포트 설정
    app.run(debug=True, host='0.0.0.0', port=80)