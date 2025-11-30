# TruthLens Backend

CNN 기반 딥페이크/이미지 위변조 자동 탐지 및 무결성 검증 웹서비스 **TruthLens**의 백엔드 저장소입니다.

## 🛠 기술 스택
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Server**: Uvicorn

## 🚀 시작하기

### 1. 환경 설정
```bash
# 가상환경 생성 (선택사항)
python -m venv venv

# Windows 가상환경 활성화
.\venv\Scripts\activate

# Mac/Linux 가상환경 활성화
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
uvicorn app.main:app --reload
```
서버는 `http://127.0.0.1:8000`에서 실행됩니다.
API 문서는 `http://127.0.0.1:8000/docs`에서 확인할 수 있습니다.

## 📚 API 명세

### Health Check
- **URL**: `GET /`
- **Description**: 서버 상태를 확인합니다.
- **Response**: `{"status": "ok", "msg": "서버 정상"}`

### 이미지 분석 (Detect)
- **URL**: `POST /api/v1/detect`
- **Description**: 이미지를 업로드하여 딥페이크 여부를 분석합니다.
- **Request**: `multipart/form-data` (Key: `file`)
- **Response**:
  ```json
  {
    "filename": "image.jpg",
    "is_fake": true,
    "score": 98.5,
    "heatmap": "base64_string..."
  }
  ```

## 📂 프로젝트 구조
```
TruthLens-Backend/
├── app/
│   ├── api/            # API 라우터 및 엔드포인트
│   ├── core/           # 설정 (Config)
│   ├── models/         # 데이터베이스/ML 모델
│   ├── schemas/        # Pydantic 스키마
│   ├── services/       # 비즈니스 로직
│   └── main.py         # 앱 진입점
├── requirements.txt    # 의존성 목록
└── README.md           # 프로젝트 설명
```
