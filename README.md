# △ BTC·ETH 4시간 롱·숏 안내 (웹 배포용)

실행: `streamlit run app.py`

## Streamlit Community Cloud 배포
1. 이 폴더의 파일을 GitHub **비공개** 저장소에 그대로 올림 (state, .streamlit 폴더 포함).
2. share.streamlit.io → Create app → 저장소 선택 → Main file `app.py` → Deploy.
3. 앱 오른쪽 위 Share에서 내 이메일 초대 → 그 주소로 접속.

- 공개 시세만 사용(API 키 불필요). 매매 일지는 서버 재시작 시 지워짐.
- 시세: 비트겟 → 막히면 OKX → 그것도 막히면 코인베이스 현물로 자동 전환. 화면 맨 아래 '△ 접속 진단'에서 확인.

## 안 돌아갈 때
- 저장소 첫 화면에 app.py가 바로 보여야 함. 폴더 안에 있으면 Main file을 `btceth_web/app.py`처럼 폴더 포함해서 지정.
- requirements.txt는 app.py와 같은 곳에. state 폴더(json 5개)도 같이 올라가 있어야 함.
- .streamlit 폴더는 없어도 됨.
- 오류 내용: 앱 화면 오른쪽 아래 'Manage app' → 로그 마지막 줄.
- 연구용 코드와 전체 설명은 btceth_bot 폴더(PC용)에 있음.
