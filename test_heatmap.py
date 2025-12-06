"""
히트맵 테스트 스크립트
서버가 실행 중일 때 이 스크립트를 실행하면 히트맵 이미지를 저장합니다.

사용법:
1. 서버 실행: uvicorn app.main:app --reload
2. 다른 터미널에서: python test_heatmap.py 이미지경로.jpg
"""

import sys
import requests
import base64
from pathlib import Path


def test_heatmap(image_path: str):
    # 서버 URL
    url = "http://127.0.0.1:8000/api/v1/detect"
    
    # 이미지 파일 열기
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        
        print(f"🔍 분석 중: {image_path}")
        response = requests.post(url, files=files)
    
    if response.status_code != 200:
        print(f"❌ 오류: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    print(f"\n📊 분석 결과:")
    print(f"   파일명: {data['filename']}")
    print(f"   딥페이크 여부: {'🚨 FAKE' if data['is_fake'] else '✅ REAL'}")
    print(f"   신뢰도: {data['score']}%")
    
    # 히트맵 이미지 저장
    if data.get("heatmap"):
        heatmap_bytes = base64.b64decode(data["heatmap"])
        output_path = "heatmap_result.png"
        
        with open(output_path, "wb") as f:
            f.write(heatmap_bytes)
        
        print(f"\n🖼️ 히트맵 이미지 저장됨: {output_path}")
        print("   (이 파일을 열어서 조작 의심 영역을 확인하세요)")
    else:
        print("\n⚠️ 히트맵이 생성되지 않았습니다.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_heatmap.py 이미지경로.jpg")
        print("예시: python test_heatmap.py test_image.jpg")
        sys.exit(1)
    
    test_heatmap(sys.argv[1])
