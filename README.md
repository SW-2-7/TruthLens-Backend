# TruthLens Backend

CNN 기반 딥페이크/이미지 위변조 자동 탐지 및 무결성 검증 웹서비스 **TruthLens**의 백엔드 저장소입니다.

## ✨ 주요 기능
- 🔍 **딥페이크 탐지**: ResNet50 기반 이미지 분류
- 🎨 **Grad-CAM 히트맵**: 조작 의심 영역 시각화
- ⚡ **FastAPI**: 고성능 비동기 API

## 🛠 기술 스택
| 구분 | 기술 |
|------|------|
| Language | Python 3.10+ |
| Framework | FastAPI |
| ML | PyTorch, ResNet50 |
| Server | Uvicorn |

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
- 서버: `http://127.0.0.1:8000`
- API 문서: `http://127.0.0.1:8000/docs`

## 📚 API 명세

### Health Check
| 항목 | 내용 |
|------|------|
| URL | `GET /` |
| Response | `{"status": "ok", "msg": "서버 정상"}` |

### 이미지 분석
| 항목 | 내용 |
|------|------|
| URL | `POST /api/v1/detect` |
| Content-Type | `multipart/form-data` |
| 파일 크기 제한 | 최대 30MB |
| 지원 형식 | JPG, PNG, GIF, WEBP |

**Response:**
```json
{
  "filename": "image.jpg",
  "is_fake": true,
  "score": 98.5,
  "heatmap": "base64_encoded_image..."
}
```

### 에러 응답
| 상태 코드 | 설명 |
|-----------|------|
| 400 | 지원하지 않는 파일 형식 |
| 413 | 파일 크기 초과 (30MB) |

## 📂 프로젝트 구조
```
TruthLens-Backend/
├── app/
│   ├── api/endpoints/   # API 엔드포인트
│   ├── core/            # 설정
│   ├── model/           # AI 모델 (ResNet50, Grad-CAM)
│   ├── schemas/         # Pydantic 스키마
│   └── main.py          # 앱 진입점
├── requirements.txt
└── README.md
```

## 🧪 테스트
```bash
# 서버 실행 후 다른 터미널에서
python test_heatmap.py test_image.jpg
```
`heatmap_result.png` 파일이 생성되어 히트맵을 확인할 수 있습니다.
