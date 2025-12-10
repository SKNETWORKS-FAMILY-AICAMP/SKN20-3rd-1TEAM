"""
온통청년 API를 통한 정책 데이터 수집
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
YOUTH_POLICY_API = os.getenv('YOUTH_POLICY_API')

def fetch_youth_policies(page_size):
    """
    온통청년 API를 통해 정책 데이터 가져오기
    
    Args:
        page_size (int): 가져올 정책 개수
    
    Returns:
        dict: API 응답 데이터
    """
    # 여러 가능한 엔드포인트 시도
    endpoints = [
        {
            'url': "https://www.youthcenter.go.kr/go/ythip/getPlcy",
            'params': {
                'apiKeyNm': YOUTH_POLICY_API,
                'pageSize': page_size,
            }
        }
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    for i, endpoint in enumerate(endpoints, 1):
        api_url = endpoint['url']
        params = endpoint['params']
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=60)
            
            
            if response.status_code == 200:
                # JSON 파싱
                try:
                    data = response.json()
                    return data
                    
                except json.JSONDecodeError as e:
                    continue
            else:
                continue
                
        except requests.exceptions.RequestException as e:
            continue
    
    return None


def save_json(data, filename="youth_policies_api"):
    """
    데이터를 JSON 파일로 저장
    """
    # 현재 스크립트 위치에서 data/raw 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(raw_dir, f"{filename}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath


def main():
    # API 호출
    data = fetch_youth_policies(page_size=3550)
    
    if data:
        # JSON 저장
        filepath = save_json(data)


if __name__ == "__main__":
    main()
