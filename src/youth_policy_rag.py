# %% [markdown]
# ## 1. 환경 설정 및 라이브러리 임포트

# %%
import os
import warnings
import logging
warnings.filterwarnings('ignore')

from dotenv import load_dotenv

# 로깅 설정
# DEBUG: 개발 중 상세 정보 확인
# INFO: 일반 실행 정보
# WARNING: 경고 (기본 Streamlit 배포시 권장)
# ERROR: 오류만 출력
logging.basicConfig(
    level=logging.INFO,  # 개발: DEBUG, 배포: WARNING
    format='%(message)s'  # 간단한 포맷 (시간/레벨 생략)
)
logger = logging.getLogger('youth_policy_rag')

# 환경변수 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.')

logger.info('✅ API 키 설정 완료')

# 실시간 현재 시간 확인
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info(f'✅ 현재 시간: {current_time}')

# %%
# LangChain 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger.info('✅ LangChain 라이브러리 임포트 완료')

# %%
# 날짜 필터링 유틸리티 함수
from datetime import datetime
import re

def parse_date_range(date_str: str) -> tuple:
    """
    신청기간 문자열을 파싱하여 (시작일, 종료일) 튜플 반환
    예: '20251125 ~ 20251204' -> (datetime(2025,11,25), datetime(2025,12,4))
    
    Returns:
        tuple: (start_date, end_date) or (None, None) if parsing fails
    """
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return (None, None)
    
    try:
        # '~' 또는 '-'로 분리
        parts = re.split(r'\s*[~\-]\s*', date_str.strip())
        
        if len(parts) == 2:
            start_str, end_str = parts
            # YYYYMMDD 형식 파싱
            start_date = datetime.strptime(start_str.strip(), '%Y%m%d')
            end_date = datetime.strptime(end_str.strip(), '%Y%m%d')
            return (start_date, end_date)
        elif len(parts) == 1:
            # 단일 날짜인 경우
            single_date = datetime.strptime(parts[0].strip(), '%Y%m%d')
            return (single_date, single_date)
    except (ValueError, AttributeError):
        pass
    
    return (None, None)


def is_currently_available(date_str: str, today: datetime = None) -> bool:
    """
    현재 신청 가능한지 확인 (1단계)
    - 빈 문자열/N/A는 상시 모집으로 간주 → True
    - 오늘 날짜가 신청기간 내에 있으면 True
    """
    if today is None:
        today = datetime.now()
    
    # 빈 값은 상시 모집으로 간주
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return True
    
    start_date, end_date = parse_date_range(date_str)
    
    # 파싱 실패시 상시 모집으로 간주
    if start_date is None or end_date is None:
        return True
    
    # 오늘이 신청기간 내인지 확인
    return start_date <= today <= end_date


def is_this_year(date_str: str, year: int = None) -> bool:
    """
    올해 진행 중인 정책인지 확인 (2단계)
    - 빈 문자열/N/A는 상시 모집으로 간주 → True
    - 신청기간이 올해에 걸쳐있으면 True
    """
    if year is None:
        year = datetime.now().year
    
    # 빈 값은 상시 모집으로 간주
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return True
    
    start_date, end_date = parse_date_range(date_str)
    
    # 파싱 실패시 포함
    if start_date is None or end_date is None:
        return True
    
    # 신청기간이 올해에 걸쳐있는지 확인
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31)
    
    # 기간이 올해와 겹치는지 확인
    return not (end_date < year_start or start_date > year_end)


def filter_docs_by_time(docs: list, time_mode: int, today: datetime = None) -> list:
    """
    시간 기반으로 문서 필터링
    
    Args:
        docs: 검색된 문서 리스트
        time_mode: 1(현재 신청 가능), 2(올해), 3(전체)
        today: 기준 날짜 (기본: 현재)
    
    Returns:
        list: 필터링된 문서 리스트
    """
    if today is None:
        today = datetime.now()
    
    if time_mode == 3:  # 전체 - 필터링 없음
        return docs
    
    filtered = []
    for doc in docs:
        date_str = doc.metadata.get('신청기간', '')
        
        if time_mode == 1:  # 현재 신청 가능
            if is_currently_available(date_str, today):
                filtered.append(doc)
        elif time_mode == 2:  # 올해 진행
            if is_this_year(date_str, today.year):
                filtered.append(doc)
    
    return filtered


logger.debug('✅ 날짜 필터링 유틸리티 함수 정의 완료')
logger.debug('   - parse_date_range(): 날짜 문자열 파싱')
logger.debug('   - is_currently_available(): 현재 신청 가능 여부')
logger.debug('   - is_this_year(): 올해 진행 여부')
logger.debug('   - filter_docs_by_time(): 시간 기반 문서 필터링')

# %%
# LLM 기반 의도 분류기 (시간 범위 판단)
INTENT_CLASSIFIER_PROMPT = '''당신은 청년 정책 질문의 의도를 분류하는 분류기입니다.
사용자의 질문을 분석하여 어떤 시간 범위의 정책을 원하는지 판단하세요.

## 분류 기준
1: 현재 신청 가능한 정책 (기본값)
   - 일반적인 정책 질문 ("취업 지원 뭐 있어?", "주거 정책 알려줘")
   - "지금", "현재", "신청 가능한" 등의 키워드
   
2: 올해 진행 중인 정책
   - "올해", "이번 년도", "2025년", "금년" 등의 키워드
   - "올해 뭐 있었어?", "이번 년도 정책" 등
   
3: 시간 무관 (역대 전체)
   - "전체", "모든", "역대", "지금까지", "과거", "종료된 것도"
   - "다 알려줘", "전부", "예전 정책도"

## 규칙
- 명확한 시간 키워드가 없으면 1(현재 신청 가능)로 분류
- 숫자만 응답하세요 (1, 2, 또는 3)

질문: {question}

분류 (1/2/3):'''


def classify_time_intent(question: str, llm) -> int:
    """
    LLM을 사용하여 질문의 시간 범위 의도 분류
    
    Args:
        question: 사용자 질문
        llm: ChatOpenAI 인스턴스
    
    Returns:
        int: 1(현재), 2(올해), 3(전체)
    """
    prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFIER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({'question': question})
        # 숫자만 추출
        time_mode = int(result.strip()[0])
        if time_mode in [1, 2, 3]:
            return time_mode
    except (ValueError, IndexError):
        pass
    
    # 기본값: 현재 신청 가능
    return 1


logger.debug('✅ LLM 기반 의도 분류기 정의 완료')
logger.debug('   - classify_time_intent(): 질문에서 시간 범위 의도 추출')

# %%
# ===== 하이브리드 전략: 질문 유형 분류 + LLM 쿼리 리라이팅 =====

# LLM 기반 질문 유형 분류기 (인사/무관/정책 3분류)
QUESTION_TYPE_CLASSIFIER_PROMPT = '''당신은 청년 정책 챗봇의 질문 유형을 분류하는 분류기입니다.
사용자의 메시지를 분석하여 어떤 유형인지 판단하세요.

## 분류 기준
1: 인사/일상 대화 (GREETING)
   - 인사: "안녕", "하이", "반가워", "안녕하세요"
   - 감사: "고마워", "감사합니다"
   - 작별: "잘가", "바이", "또 볼게"
   - 안부: "잘 지내?", "뭐해?", "좋은 하루"
   
2: 정책과 무관한 질문 (IRRELEVANT)
   - 날씨, 음식, 연예인, 게임 등 정책과 전혀 무관한 주제
   - 예: "오늘 날씨 어때?", "맛집 추천해줘", "BTS 좋아해?"
   - 단, 생활 어려움 표현은 3번으로 분류 (아래 참고)
   
3: 정책 관련 질문 (POLICY)
   - 직접적 정책 질문: "취업 지원", "주거 정책", "창업 지원" 등
   - 간접적 표현 (생활 어려움 → 정책 연결 가능):
     * "배고프다", "돈이 없어", "생활이 힘들어" → 생계/복지 정책
     * "집이 없어", "월세가 비싸", "잘곳이 없어" → 주거 정책
     * "취업이 안 돼", "일자리가 없어" → 일자리 정책
     * "학비가 부담돼", "공부하고 싶어" → 교육 정책
     * "우울해", "힘들어", "스트레스" → 복지/건강 정책
   - 청년의 어려움이나 목표와 관련된 표현은 모두 3번

## 규칙
- 명확한 인사/작별/감사만 1번
- 정책과 완전히 무관한 주제만 2번
- 조금이라도 청년의 생활/어려움/목표와 연결되면 3번
- 애매하면 3번 (정책 연결 시도)
- 숫자만 응답하세요 (1, 2, 또는 3)

메시지: {message}

분류 (1/2/3):'''


def classify_question_type(message: str, llm) -> int:
    """
    질문 유형 분류 (인사/무관/정책)
    
    Args:
        message: 사용자 메시지
        llm: ChatOpenAI 인스턴스
    
    Returns:
        int: 1(인사), 2(무관), 3(정책)
    """
    prompt = ChatPromptTemplate.from_template(QUESTION_TYPE_CLASSIFIER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({'message': message})
        question_type = int(result.strip()[0])
        if question_type in [1, 2, 3]:
            logger.debug(f'📋 질문 유형 분류: {question_type} (1:인사, 2:무관, 3:정책)')
            return question_type
    except (ValueError, IndexError) as e:
        logger.warning(f'⚠️ 질문 유형 분류 실패: {e}')
    
    # 기본값: 정책 관련 (연결 시도)
    return 3


# ===== LLM 기반 쿼리 리라이팅 (핵심!) =====
QUERY_REWRITE_PROMPT = '''당신은 청년 정책 검색 쿼리를 최적화하는 전문가입니다.
사용자의 메시지를 청년 정책 데이터베이스 검색에 적합한 키워드로 변환해주세요.

## 변환 규칙
1. 간접적/감정적 표현 → 정책 키워드로 변환
   - "배고파", "밥 먹기 힘들어" → "생계 지원 복지 수당"
   - "돈이 없어", "생활이 힘들어" → "생계 지원 복지 수당 금융"
   - "집이 없어", "잘곳이 없어", "월세가 비싸" → "주거 지원 임대 전월세"
   - "취업이 안 돼", "일자리가 없어" → "취업 지원 일자리 채용"
   - "학비가 부담돼", "공부하고 싶어" → "교육 장학금 학자금"
   - "우울해", "힘들어", "스트레스" → "마음건강 심리상담 복지"
   - "창업하고 싶어" → "창업 지원 사업자"

2. 직접적 정책 질문은 핵심 키워드만 추출
   - "서울에서 받을 수 있는 주거 지원 알려줘" → "서울 주거 지원"
   - "취업 장려금 신청하고 싶어" → "취업 장려금"

3. 검색 최적화
   - 조사, 어미 제거 (을/를, 이/가, 해줘, 알려줘 등)
   - 핵심 명사/키워드 위주로 변환
   - 2~5개 단어로 간결하게

## 출력 형식
- 변환된 검색 쿼리만 출력 (설명 없이)
- 한 줄로 출력

사용자 메시지: {message}

검색 쿼리:'''


def rewrite_query_for_search(message: str, llm) -> str:
    """
    LLM을 사용하여 사용자 메시지를 정책 검색에 적합한 쿼리로 변환
    
    Args:
        message: 원본 사용자 메시지
        llm: ChatOpenAI 인스턴스
    
    Returns:
        str: 검색에 최적화된 쿼리
    """
    prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        rewritten = chain.invoke({'message': message}).strip()
        logger.info(f'🔄 쿼리 리라이팅: "{message}" → "{rewritten}"')
        return rewritten
    except Exception as e:
        logger.warning(f'⚠️ 쿼리 리라이팅 실패: {e}, 원본 사용')
        return message


logger.debug('✅ 하이브리드 전략 정의 완료')
logger.debug('   - classify_question_type(): 인사(1)/무관(2)/정책(3) 분류')
logger.debug('   - rewrite_query_for_search(): LLM 기반 쿼리 리라이팅')

# %% [markdown]
# ## 1-2. 사용자 프로필 기반 필터링
# 
# 사용자가 대화 중 언급한 조건(나이, 지역, 관심분야, 취업상태)을 저장하고 필터링에 적용합니다.

# %%
# 사용자 프로필 클래스 정의
from dataclasses import dataclass, field
from typing import Optional, List
import json

@dataclass
class UserProfile:
    """사용자 조건을 저장하는 프로필 클래스"""
    age: Optional[int] = None                    # 나이
    region: Optional[str] = None                 # 지역 (시/도 단위)
    interests: List[str] = field(default_factory=list)  # 관심 분야 (대분류)
    employment_status: Optional[str] = None      # 취업상태: 미취업, 재직중, 창업, 학생 등
    
    def update(self, **kwargs):
        """프로필 부분 업데이트 (None이 아닌 값만 업데이트)"""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                if key == 'interests' and isinstance(value, str):
                    # 단일 관심분야 추가
                    if value not in self.interests:
                        self.interests.append(value)
                elif key == 'interests' and isinstance(value, list):
                    # 리스트로 관심분야 설정
                    self.interests = value
                else:
                    setattr(self, key, value)
    
    def clear(self):
        """프로필 초기화"""
        self.age = None
        self.region = None
        self.interests = []
        self.employment_status = None
    
    def is_empty(self) -> bool:
        """프로필이 비어있는지 확인"""
        return (self.age is None and 
                self.region is None and 
                len(self.interests) == 0 and 
                self.employment_status is None)
    
    def __str__(self):
        parts = []
        if self.age:
            parts.append(f"나이: {self.age}세")
        if self.region:
            parts.append(f"지역: {self.region}")
        if self.interests:
            parts.append(f"관심분야: {', '.join(self.interests)}")
        if self.employment_status:
            parts.append(f"취업상태: {self.employment_status}")
        return " | ".join(parts) if parts else "설정된 조건 없음"


logger.debug('✅ UserProfile 클래스 정의 완료')
logger.debug('   - age: 나이')
logger.debug('   - region: 지역 (시/도)')
logger.debug('   - interests: 관심분야 리스트')
logger.debug('   - employment_status: 취업상태')

# %%
# LLM 기반 사용자 조건 추출기
PROFILE_EXTRACTOR_PROMPT = '''당신은 사용자의 대화에서 개인 조건 정보를 추출하는 분석기입니다.
사용자의 메시지를 분석하여 다음 정보를 JSON 형식으로 추출하세요.

## 추출 항목
1. age (정수 또는 null): 나이
   - "25살", "스물다섯", "25세" → 25
   - 언급 없으면 null

2. region (문자열 또는 null): 지역 (시/도 단위로 정규화)
   - "인천 살아", "인천시 거주" → "인천"
   - "서울", "서울특별시" → "서울"
   - "경기도 수원" → "경기"
   - "부산광역시" → "부산"
   - "충북", "충청북도" → "충북"
   - "충남", "충청남도" → "충남"
   - "전북", "전라북도" → "전북"  
   - "전남", "전라남도" → "전남"
   - "경북", "경상북도" → "경북"
   - "경남", "경상남도" → "경남"
   - "제주", "제주도", "제주특별자치도" → "제주"
   - "강원", "강원도", "강원특별자치도" → "강원"
   - "세종", "세종시", "세종특별자치시" → "세종"
   - 언급 없으면 null

3. interest (문자열 또는 null): 관심 분야 (대분류)
   - "취업", "일자리", "직장" → "일자리"
   - "집", "전세", "월세", "주거" → "주거"
   - "공부", "학비", "장학금" → "교육"
   - "창업", "사업" → "일자리"
   - "문화", "여가", "건강", "복지" → "복지문화"
   - "권리", "참여", "정책제안" → "참여권리"
   - 언급 없으면 null

4. employment_status (문자열 또는 null): 취업 상태
   - "백수", "무직", "구직중", "취준생", "미취업" → "미취업"
   - "직장인", "회사원", "재직중" → "재직중"
   - "사장", "자영업", "창업" → "창업"
   - "대학생", "학생", "재학중" → "학생"
   - 언급 없으면 null

## 규칙
- 명확하게 언급된 정보만 추출
- 추측하지 말 것
- 반드시 JSON 형식으로만 응답

## 예시
입력: "나 25살이고 인천 살아. 취업 준비 중인데 뭐 있어?"
출력: {{"age": 25, "region": "인천", "interest": "일자리", "employment_status": "미취업"}}

입력: "주거 지원 정책 알려줘"
출력: {{"age": null, "region": null, "interest": "주거", "employment_status": null}}

입력: "서울에서 창업하려는데 지원받을 수 있는 거 있어?"
출력: {{"age": null, "region": "서울", "interest": "일자리", "employment_status": "창업"}}

사용자 메시지: {message}

JSON 출력:'''


def extract_user_profile(message: str, llm) -> dict:
    """
    사용자 메시지에서 프로필 정보 추출
    
    Args:
        message: 사용자 메시지
        llm: ChatOpenAI 인스턴스
    
    Returns:
        dict: 추출된 프로필 정보 (age, region, interest, employment_status)
    """
    prompt = ChatPromptTemplate.from_template(PROFILE_EXTRACTOR_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({'message': message})
        # JSON 파싱
        # 혹시 ```json ... ``` 형태로 응답하면 정리
        result = result.strip()
        if result.startswith('```'):
            result = result.split('```')[1]
            if result.startswith('json'):
                result = result[4:]
        result = result.strip()
        
        parsed = json.loads(result)
        return {
            'age': parsed.get('age'),
            'region': parsed.get('region'),
            'interest': parsed.get('interest'),
            'employment_status': parsed.get('employment_status')
        }
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"⚠️ 프로필 추출 실패: {e}")
        return {'age': None, 'region': None, 'interest': None, 'employment_status': None}


logger.debug('✅ 사용자 프로필 추출기 정의 완료')
logger.debug('   - extract_user_profile(): 자연어에서 나이/지역/분야/취업상태 추출')

# %%
# 사용자 프로필 기반 필터링 함수들

def is_age_eligible(doc, user_age: int) -> bool:
    """
    사용자 나이가 정책의 지원 연령 범위에 해당하는지 확인
    
    Args:
        doc: 문서 객체
        user_age: 사용자 나이
    
    Returns:
        bool: 연령 조건 충족 여부
    """
    if user_age is None:
        return True  # 나이 정보 없으면 필터링 안 함
    
    min_age = doc.metadata.get('지원최소연령', 0)
    max_age = doc.metadata.get('지원최대연령', 0)
    
    # 문자열인 경우 정수로 변환
    try:
        min_age = int(min_age) if min_age else 0
        max_age = int(max_age) if max_age else 0
    except (ValueError, TypeError):
        min_age, max_age = 0, 0
    
    # 0은 제한 없음을 의미
    if min_age == 0 and max_age == 0:
        return True  # 연령 제한 없는 정책
    
    if min_age == 0:
        return user_age <= max_age
    if max_age == 0:
        return user_age >= min_age
    
    return min_age <= user_age <= max_age


def is_region_match(doc, user_region: str) -> bool:
    """
    사용자 지역이 정책의 지원 지역과 일치하는지 확인
    
    Args:
        doc: 문서 객체
        user_region: 사용자 지역 (시/도 단위)
    
    Returns:
        bool: 지역 조건 충족 여부
    """
    if user_region is None:
        return True  # 지역 정보 없으면 필터링 안 함
    
    # 정책의 주관기관/등록기관에서 지역 확인
    org_name = doc.metadata.get('주관기관명', '') + doc.metadata.get('등록기관명', '')
    
    # 지역 매핑 (사용자 입력 → 검색 키워드)
    region_keywords = {
        '서울': ['서울'],
        '부산': ['부산'],
        '대구': ['대구'],
        '인천': ['인천'],
        '광주': ['광주'],
        '대전': ['대전'],
        '울산': ['울산'],
        '세종': ['세종'],
        '경기': ['경기'],
        '강원': ['강원'],
        '충북': ['충북', '충청북도'],
        '충남': ['충남', '충청남도'],
        '전북': ['전북', '전라북도'],
        '전남': ['전남', '전라남도'],
        '경북': ['경북', '경상북도'],
        '경남': ['경남', '경상남도'],
        '제주': ['제주'],
    }
    
    # 전국 단위 정책 키워드 (중앙부처)
    national_keywords = ['고용노동부', '보건복지부', '국토교통부', '중소벤처기업부', 
                        '교육부', '문화체육관광부', '여성가족부', '행정안전부']
    
    # 전국 단위 정책은 모두 통과
    for keyword in national_keywords:
        if keyword in org_name:
            return True
    
    # 사용자 지역과 매칭
    keywords = region_keywords.get(user_region, [user_region])
    for keyword in keywords:
        if keyword in org_name:
            return True
    
    return False


def is_interest_match(doc, interests: list) -> bool:
    """
    사용자 관심분야가 정책 분야와 일치하는지 확인
    
    Args:
        doc: 문서 객체
        interests: 사용자 관심분야 리스트
    
    Returns:
        bool: 분야 조건 충족 여부
    """
    if not interests:
        return True  # 관심분야 없으면 필터링 안 함
    
    policy_category = doc.metadata.get('대분류', '') + doc.metadata.get('중분류', '')
    
    for interest in interests:
        if interest in policy_category:
            return True
    
    return False


def is_employment_match(doc, employment_status: str) -> bool:
    """
    사용자 취업상태가 정책 조건과 일치하는지 확인
    
    Args:
        doc: 문서 객체
        employment_status: 사용자 취업상태
    
    Returns:
        bool: 취업상태 조건 충족 여부
    """
    if employment_status is None:
        return True  # 취업상태 없으면 필터링 안 함
    
    # 정책의 참여제외대상, 추가자격조건 확인
    exclude_target = doc.metadata.get('참여제외대상', '')
    requirements = doc.metadata.get('추가자격조건', '')
    policy_content = doc.page_content
    
    # 취업상태별 매칭 로직
    if employment_status == '미취업':
        # 재직자 전용 정책 제외
        if '재직' in requirements and '미취업' not in requirements:
            return False
        return True
    
    elif employment_status == '재직중':
        # 미취업자 전용 정책 제외
        if '미취업' in requirements and '재직' not in requirements:
            return False
        return True
    
    elif employment_status == '창업':
        # 창업 관련 정책 우대
        if '창업' in policy_content or '사업자' in policy_content:
            return True
        return True  # 일단 모든 정책 포함
    
    elif employment_status == '학생':
        # 학생 관련 정책 우대
        if '대학생' in exclude_target or '재학생' in exclude_target:
            return False
        return True
    
    return True


def filter_docs_by_profile(docs: list, profile: 'UserProfile') -> list:
    """
    사용자 프로필 기반으로 문서 필터링 (모든 조건 AND)
    
    Args:
        docs: 검색된 문서 리스트
        profile: UserProfile 객체
    
    Returns:
        list: 필터링된 문서 리스트
    """
    if profile is None or profile.is_empty():
        return docs  # 프로필 없으면 필터링 안 함
    
    filtered = []
    for doc in docs:
        # 모든 조건을 AND로 확인
        if (is_age_eligible(doc, profile.age) and
            is_region_match(doc, profile.region) and
            is_interest_match(doc, profile.interests) and
            is_employment_match(doc, profile.employment_status)):
            filtered.append(doc)
    
    return filtered


logger.debug('✅ 사용자 프로필 기반 필터링 함수 정의 완료')
logger.debug('   - is_age_eligible(): 연령 조건 확인')
logger.debug('   - is_region_match(): 지역 조건 확인')
logger.debug('   - is_interest_match(): 관심분야 조건 확인')
logger.debug('   - is_employment_match(): 취업상태 조건 확인')
logger.debug('   - filter_docs_by_profile(): 통합 프로필 필터링')

# %% [markdown]
# ## 2. VectorDB 로드
# 
# 기존에 구축된 ChromaDB(youth_policies 컬렉션)를 로드합니다.

# %%
# 프로젝트 경로 설정
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir) if current_dir.endswith('src') else current_dir
db_path = os.path.join(project_root, 'data', 'vectordb')

logger.info(f'📁 VectorDB 경로: {db_path}')
logger.debug(f'📁 경로 존재 여부: {os.path.exists(db_path)}')

# %%
# 임베딩 모델 설정
# 모델 테스트 단계라 추후 정리 할 예정입니다.

# # 1. OpenAI 모델
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
logger.info('✅ OpenAI API 임베딩 모델 선택 완료')


# # 2.KoSimCSE-RoBERTa) (고성능)
# from langchain_community.embeddings import HuggingFaceEmbeddings
# model_name = "BM-K/KoSimCSE-roberta-large" 
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'}, # GPU가 없다면 'cpu' 사용
#     encode_kwargs={'normalize_embeddings': True}
# )
# logger.info(f'✅ 로컬 임베딩 모델 선택 완료: {model_name}')


# # 3. KCBERT-Base (범용 안정성)
# from langchain_community.embeddings import HuggingFaceEmbeddings
# model_name = "beomi/kcbert-base"
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'}, 
#     encode_kwargs={'normalize_embeddings': True}
# )
# logger.info(f'✅ 로컬 임베딩 모델 선택 완료: {model_name}')


# # 4. SKT KoBERT-SBERT (빠른 속도)
# from langchain_community.embeddings import HuggingFaceEmbeddings
# model_name = "skt/kobert-base-v1" 
# embedding_model = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs={'device': 'cpu'}, 
#     encode_kwargs={'normalize_embeddings': True}
# )
# logger.info(f'✅ 로컬 임베딩 모델 선택 완료: {model_name}')
# 기존 ChromaDB 로드
vectorstore = Chroma(
    persist_directory=db_path,
    collection_name='youth_policies',
    embedding_function=embedding_model
)

policy_count = vectorstore._collection.count()
logger.info(f'✅ VectorDB 로드 완료')
logger.info(f'📊 저장된 정책 수: {policy_count}개')

# %% [markdown]
# ## 3. RAG 체인 구성
# 
# 리트리버, 프롬프트, LLM을 연결하여 RAG 체인을 구성합니다.

# %%
# 리트리버 설정
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

logger.info('✅ 리트리버 설정 완료 (top_k=5)')

# %%
# 문서 포맷팅 함수
def format_docs(docs):
    """검색된 문서를 문자열로 포맷팅"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        text = f"""
[정책 {i}]
정책명: {metadata.get('정책명', 'N/A')}
분야: {metadata.get('대분류', '')} > {metadata.get('중분류', '')}
주관기관: {metadata.get('주관기관명', 'N/A')}
신청기간: {metadata.get('신청기간', 'N/A')}
신청방법: {metadata.get('신청방법', 'N/A')}
지원연령: {metadata.get('지원최소연령', '0')}세 ~ {metadata.get('지원최대연령', '0')}세
지원금액: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원
신청URL: {metadata.get('신청URL', 'N/A')}

내용:
{doc.page_content}
"""
        formatted.append(text)
    return '\n\n---\n\n'.join(formatted)

logger.debug('✅ 문서 포맷팅 함수 정의 완료')

# %%
# LLM 설정
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

logger.info('✅ LLM 설정 완료 (gpt-4o-mini)')

# %%
# 프롬프트 템플릿 정의 (전역 변수로 관리 - 클래스에서도 재사용)

# ===== 인사/일상 대화용 프롬프트 =====
GREETING_PROMPT = '''당신은 "청년이음"의 스마트한 선배입니다.
친근하고 따뜻한 말투로 대화해주세요.

## 페르소나
- 이름: 스마트한 선배 (Smart Mentor)
- 성격: 친절하고 격려하며, 편안한 분위기
- 말투: 대학 선배가 후배에게 말하듯 편안하고 친근하게 (반말 사용)

## 규칙
1. 따뜻하고 친근하게 인사에 응답하세요.
2. 간단히 자기소개를 하고, 청년 정책 관련 질문을 유도하세요.
3. 1~3문장 정도로 짧게 응답하세요.

## 응답 예시
- "안녕! 나는 청년이음의 스마트한 선배야 😊 취업, 주거, 창업 등 청년 정책 궁금한 거 있으면 편하게 물어봐!"
- "반가워! 오늘도 좋은 하루 보내고 있어? 청년 정책 관련해서 도움 줄 수 있는 거 있으면 말해줘! 💪"
- "하이! 나는 청년들을 위한 정책 안내 도우미야. 뭐든 궁금한 거 있으면 물어봐~"

사용자 메시지: {message}

응답:'''


# ===== 무관 질문 거절용 프롬프트 =====
IRRELEVANT_PROMPT = '''당신은 "청년이음"의 스마트한 선배입니다.
정책과 무관한 질문에 친절하게 거절하고, 챗봇의 목적을 안내해주세요.

## 페르소나
- 말투: 대학 선배가 후배에게 말하듯 편안하고 친근하게 (반말 사용)

## 규칙
1. 딱딱하게 거절하지 말고, 선배처럼 친근하게 안내하세요.
2. 청년 정책 챗봇임을 알려주고, 도움 줄 수 있는 분야를 안내하세요.
3. 2~3문장 정도로 응답하세요.

## 도움 가능 분야 (안내용)
- 🏢 일자리: 취업 지원, 인턴십, 창업 지원
- 🏠 주거: 전월세 지원, 청년 주택
- 🎓 교육: 장학금, 학자금 대출
- 🎨 복지: 청년 수당, 마음 건강 지원
- ⚖️ 참여: 정책 제안, 청년 공간

## 응답 예시
- "아 그건 내 전문 분야가 아니라서 잘 모르겠어 ㅋㅋ 나는 청년 정책 전문이거든! 취업, 주거, 교육, 복지 관련해서는 뭐든 물어봐 😊"
- "음, 그건 대답하기 어렵네! 나는 청년 정책 안내하는 챗봇이야. 일자리, 집, 장학금 같은 거 궁금하면 도와줄게! 💪"

사용자 메시지: {message}

응답:'''


# ===== 정책 RAG용 시스템 프롬프트 (추론 규칙 추가) =====
SYSTEM_PROMPT = '''당신은 "청년이음"의 스마트한 선배입니다.
복잡한 청년 정책을 후배에게 쉽고 친근하게 설명해주는 역할입니다.

## 현재 시간 정보
- 오늘 날짜: {today}

## 페르소나
- 이름: 스마트한 선배 (Smart Mentor)
- 성격: 친절하고 격려하며, 명확하게 설명함
- 말투: 대학 선배가 후배에게 알려주듯 편안하고 이해하기 쉽게 (반말 사용)

## 🔥 추론적 의도 분석 (중요!)
사용자가 직접 정책 키워드를 말하지 않아도, 생활의 어려움이나 목표와 관련된 표현이면 연관 정책을 추천하세요:
- "배고프다", "돈이 없어", "생활이 힘들어" → 생계/복지 지원 정책 추천
- "집이 없어", "월세가 너무 비싸" → 주거 지원 정책 추천
- "취업이 안 돼", "일자리가 없어" → 일자리/취업 지원 정책 추천
- "학비가 부담돼", "공부하고 싶어" → 교육/장학금 정책 추천
- "우울해", "지쳐", "스트레스 받아" → 청년 마음건강/복지 정책 추천
- "창업하고 싶어", "사업 아이디어가 있어" → 창업 지원 정책 추천

## 답변 형식 (반드시 이 구조를 따르세요)

### 1️⃣ 인사/공감 (첫 1~2문장)
밝고 긍정적인 분위기로 시작하세요.
예시:
- "요즘 이 정책 궁금해하는 사람 많더라구!"
- "오, 좋은 질문이야! 도움 될 만한 정보가 있어서 공유하고 싶었어!"
- "이거 진짜 알아두면 좋은 정책이야!"
- "힘든 상황이구나... 도움 될 만한 정책 찾아봤어!" (간접 표현 시)

### 2️⃣ 정책 핵심 요약
"{{정책명}}은(는) {{정책 목적/핵심 요약}}에 도움이 되는 정책이야." 형식으로 설명하세요.

### 3️⃣ 구조화된 안내 (이모지 포함)
아래 형식을 사용하세요:
✨ 지원 내용: {{지원내용}} 
📝 신청 자격: {{자격조건}} 
🗓️ 신청 기간: {{신청기간 - 제공된 원본 데이터 그대로 표시}} 
🧾 신청 방법: {{신청방법}} 
🔗 신청하러 가기: {{URL}} (있을 경우만)

### 4️⃣ 마무리 (밝고 권유형)
예시:
- "조건 맞으면 꼭 한 번 확인해보는 거 추천해!"
- "이건 진짜 놓치기 아쉬운 정책이야."
- "해당되면 바로 신청해봐!"

### 5️⃣ 추가 안내
마지막에 항상 이 문장을 포함하세요:
"더 궁금한 거 있으면 편하게 물어봐! 😊"

## 정책 미검색 시 안내 (중요!)
검색된 정책이 없거나 적합한 정책을 찾지 못한 경우, 딱딱하게 거절하지 말고 다음과 같이 안내하세요:
"음, 딱 맞는 정책을 찾기가 좀 어렵네! 혹시 이런 분야에 관심 있어?
- 🏢 일자리: 취업 지원, 인턴, 창업
- 🏠 주거: 전월세, 청년 주택
- 🎓 교육: 장학금, 학자금
- 🎨 복지: 청년 수당, 마음 건강
좀 더 구체적으로 알려주면 더 잘 찾아볼게! 😊"

## 답변 규칙
1. 제공된 정책 정보만을 기반으로 답변하세요.
2. 어려운 행정 용어는 쉽게 풀어서 설명하세요.
3. 여러 정책이 있으면 각각 위 형식으로 안내하세요.
4. 반말을 사용하되 존중하는 톤을 유지하세요.
5. "총 몇 개", "전체 개수" 등 총 정책 수를 묻는 질문에는 반드시 제공된 [총 정책 수]를 사용해서 답변하세요.
6. 사용자가 특정 개수(예: 6개, 10개)를 요청하면 제공된 정책 중에서 해당 개수만큼만 안내하세요.
7. 신청기간은 제공된 원본 데이터를 그대로 표시하세요. (예: "20251125 ~ 20251204", "N/A" 등)
8. 신청기간이 "N/A"이거나 비어있으면 "상시 모집 또는 별도 공지"라고 안내하세요.

## 정책 분야 참고
- 🏢 일자리: 취업 장려금, 창업 지원, 직무 교육, 인턴십
- 🏠 주거: 전월세 보증금 대출, 월세 지원, 공공임대
- 🎓 교육: 장학금, 학자금 대출, 자격증 응시료 지원
- 🎨 복지/문화: 청년 수당, 마음 건강 지원, 문화 바우처
- ⚖️ 참여/권리: 정책 제안, 청년 공간, 법률 상담
'''

HUMAN_PROMPT = '''
[검색된 관련 정책 정보]:
{context}

질문: {question}

답변:'''


# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM_PROMPT),
    ('human', HUMAN_PROMPT)
])

logger.debug('✅ 프롬프트 템플릿 정의 완료')
logger.debug('   - GREETING_PROMPT: 인사/일상 대화용')
logger.debug('   - IRRELEVANT_PROMPT: 무관 질문 거절용')
logger.debug('   - SYSTEM_PROMPT: 정책 RAG용 (추론 규칙 포함)')

# %%
# LCEL 방식 RAG 체인 구성
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

logger.info('✅ RAG 체인 구성 완료')

# %% [markdown]
# ## 4. RAG 클래스 정의
# 
# 사용하기 편리하도록 RAG 시스템을 클래스로 래핑합니다.

# %%
class YouthPolicyRAG:
    """청년 정책 RAG 시스템 클래스 (하이브리드 전략 + LLM 쿼리 리라이팅)"""
    
    def __init__(self, db_path, retriever_k=5, search_k_multiplier=4, default_time_mode=1, max_history=20):
        """
        RAG 시스템 초기화
        
        Args:
            db_path: VectorDB 경로
            retriever_k: 최종 반환할 문서 개수
            search_k_multiplier: 필터링 전 검색 배수 (retriever_k * multiplier 만큼 검색)
            default_time_mode: 기본 시간 모드 (1: 현재 신청 가능)
            max_history: 대화 기록 최대 저장 개수 (기본: 20)
        """
        logger.info('=' * 60)
        logger.info('🚀 청년 정책 RAG 시스템 초기화')
        logger.info('   💡 하이브리드 전략: 인사/무관/정책 질문 분류')
        logger.info('   💡 LLM 쿼리 리라이팅: 간접 표현 → 정책 키워드 변환')
        logger.info('   💡 사용자 프로필 기반 맞춤 필터링 지원')
        logger.info('=' * 60)
        
        # 임베딩 모델
        self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # VectorDB 로드
        self.vectorstore = Chroma(
            persist_directory=db_path,
            collection_name='youth_policies',
            embedding_function=self.embedding_model
        )
        self.total_policies = self.vectorstore._collection.count()
        logger.info(f'✅ VectorDB 로드 완료 (정책 수: {self.total_policies}개)')
        
        # 검색 설정
        self.retriever_k = retriever_k
        self.search_k = retriever_k * search_k_multiplier
        self.default_time_mode = default_time_mode
        
        # LLM 설정
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        # 사용자 프로필 (세션 유지 - 휘발성)
        self.user_profile = UserProfile()
        
        # 대화 기록 (세션 유지 - 휘발성)
        self.chat_history = []  # [{"role": "user"|"assistant", "content": "..."}, ...]
        self.max_history = max_history
        
        # 초기화 상태 플래그 (Streamlit 대응)
        self.is_initialized = True
        
        logger.info(f'✅ 검색 설정: 선 검색 {self.search_k}개 → 시간 필터링 → 프로필 필터링 → {self.retriever_k}개 반환')
        logger.info(f'✅ 대화 기록: 최대 {self.max_history}개 저장')
        logger.info('✅ RAG 시스템 준비 완료!\n')
    
    # ===== 하이브리드 전략: 질문 유형별 응답 =====
    
    def _handle_greeting(self, message: str) -> str:
        """인사/일상 대화 처리 (RAG 없이 LLM만 사용)"""
        prompt = ChatPromptTemplate.from_template(GREETING_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({'message': message})
    
    def _handle_irrelevant(self, message: str) -> str:
        """무관 질문 처리 (친절한 거절 + 안내)"""
        prompt = ChatPromptTemplate.from_template(IRRELEVANT_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({'message': message})
    
    def _handle_policy_question(self, question: str, time_mode: int = None) -> tuple:
        """
        정책 관련 질문 처리 (RAG 모드 + LLM 쿼리 리라이팅)
        
        Returns:
            tuple: (answer, docs)
        """
        # 사용자 프로필 추출 및 업데이트
        self._extract_and_update_profile(question)
        
        # 시간 모드 결정
        if time_mode is None:
            time_mode = self._classify_intent(question)
        
        # ★ LLM 기반 쿼리 리라이팅 (간접 표현 → 정책 키워드)
        search_query = rewrite_query_for_search(question, self.llm)
        
        # 검색 및 필터링
        docs = self._search_and_filter(search_query, time_mode)
        
        # 컨텍스트 생성
        context = self._format_docs(docs)
        
        # 프롬프트 생성 및 답변
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        prompt = ChatPromptTemplate.from_messages([
            ('system', SYSTEM_PROMPT),
            ('human', HUMAN_PROMPT)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            'context': context,
            'question': question,  # 원본 질문 사용 (LLM이 자연스럽게 응답하도록)
            'today': today_str
        })
        
        return answer, docs
    
    # ===== 기존 내부 메서드 =====
    
    def _classify_intent(self, question: str) -> int:
        """LLM을 사용하여 시간 범위 의도 분류"""
        return classify_time_intent(question, self.llm)
    
    def _extract_and_update_profile(self, message: str) -> dict:
        """
        메시지에서 사용자 프로필 정보를 추출하고 업데이트
        
        Returns:
            dict: 추출된 프로필 정보 (새로 추출된 것만)
        """
        extracted = extract_user_profile(message, self.llm)
        
        # 프로필 업데이트 (None이 아닌 값만)
        if extracted['age'] is not None:
            self.user_profile.age = extracted['age']
        if extracted['region'] is not None:
            self.user_profile.region = extracted['region']
        if extracted['interest'] is not None:
            if extracted['interest'] not in self.user_profile.interests:
                self.user_profile.interests.append(extracted['interest'])
        if extracted['employment_status'] is not None:
            self.user_profile.employment_status = extracted['employment_status']
        
        return extracted
    
    def _search_and_filter(self, query: str, time_mode: int, top_k: int = None, use_profile: bool = True) -> list:
        """
        검색 후 시간 + 프로필 기반 필터링 수행 (핵심 로직)
        
        Args:
            query: 검색 쿼리
            time_mode: 1(현재), 2(올해), 3(전체)
            top_k: 반환할 문서 수 (기본: self.retriever_k)
            use_profile: 사용자 프로필 필터링 적용 여부
        
        Returns:
            list: 필터링된 문서 리스트
        """
        if top_k is None:
            top_k = self.retriever_k
        
        # 더 많은 문서를 먼저 검색
        docs = self.vectorstore.similarity_search(query, k=self.search_k)
        logger.debug(f'🔍 VectorDB 검색: "{query}" → {len(docs)}개 결과')
        
        # 1단계: 시간 기반 필터링
        filtered_docs = filter_docs_by_time(docs, time_mode)
        logger.debug(f'📅 시간 필터링 후: {len(filtered_docs)}개')
        
        # 2단계: 사용자 프로필 기반 필터링 (옵션)
        if use_profile and not self.user_profile.is_empty():
            filtered_docs = filter_docs_by_profile(filtered_docs, self.user_profile)
            logger.debug(f'👤 프로필 필터링 후: {len(filtered_docs)}개')
        
        # top_k 개수만큼 반환
        return filtered_docs[:top_k]
    
    def _format_docs(self, docs: list) -> str:
        """검색된 문서를 포맷팅 (원본 데이터 그대로)"""
        formatted = []
        
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            
            text = f"""
[정책 {i}]
정책명: {metadata.get('정책명', 'N/A')}
분야: {metadata.get('대분류', '')} > {metadata.get('중분류', '')}
주관기관: {metadata.get('주관기관명', 'N/A')}
신청기간: {metadata.get('신청기간', 'N/A')}
신청방법: {metadata.get('신청방법', 'N/A')}
지원연령: {metadata.get('지원최소연령', '0')}세 ~ {metadata.get('지원최대연령', '0')}세
지원금액: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원
신청URL: {metadata.get('신청URL', 'N/A')}

내용:
{doc.page_content}
"""
            formatted.append(text)
        return '\n\n---\n\n'.join(formatted)
    
    def _add_to_history(self, role: str, content: str):
        """
        대화 기록에 메시지 추가
        
        Args:
            role: "user" 또는 "assistant"
            content: 메시지 내용
        """
        self.chat_history.append({"role": role, "content": content})
        
        # 최대 개수 초과시 오래된 것부터 삭제
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]
    
    # ===== 공개 API 메서드 (하이브리드 전략 적용) =====
    
    def ask(self, question: str, time_mode: int = None) -> str:
        """
        질문에 대한 답변 생성 (하이브리드 전략 적용)
        
        Args:
            question: 사용자 질문
            time_mode: 시간 모드 (None이면 LLM이 자동 판단)
        
        Returns:
            str: 답변 문자열
        """
        # 1단계: 질문 유형 분류
        question_type = classify_question_type(question, self.llm)
        logger.info(f'📌 질문 유형: {question_type} (1:인사, 2:무관, 3:정책)')
        
        # 2단계: 유형별 처리
        if question_type == 1:  # 인사/일상 대화
            answer = self._handle_greeting(question)
        elif question_type == 2:  # 정책과 무관
            answer = self._handle_irrelevant(question)
        else:  # 정책 관련 (기본값)
            answer, _ = self._handle_policy_question(question, time_mode)
        
        # 대화 기록 저장
        self._add_to_history("user", question)
        self._add_to_history("assistant", answer)
        
        return answer
    
    def ask_with_sources(self, question: str, time_mode: int = None) -> dict:
        """
        질문에 대한 답변과 출처 반환 (하이브리드 전략 적용)
        
        Args:
            question: 사용자 질문
            time_mode: 시간 모드 (None이면 LLM이 자동 판단)
        
        Returns:
            dict: {
                'answer': 답변 문자열,
                'sources': 참고 정책 리스트,
                'retrieved_count': 검색된 정책 수,
                'user_profile': 현재 프로필 (dict),
                'question_type': 질문 유형 (1:인사, 2:무관, 3:정책)
            }
        """
        # 1단계: 질문 유형 분류
        question_type = classify_question_type(question, self.llm)
        logger.info(f'📌 질문 유형: {question_type} (1:인사, 2:무관, 3:정책)')
        
        # 2단계: 유형별 처리
        sources = []
        docs = []
        
        if question_type == 1:  # 인사/일상 대화
            answer = self._handle_greeting(question)
        elif question_type == 2:  # 정책과 무관
            answer = self._handle_irrelevant(question)
        else:  # 정책 관련 (기본값)
            answer, docs = self._handle_policy_question(question, time_mode)
            
            # 출처 정보 추출
            for doc in docs:
                sources.append({
                    '정책명': doc.metadata.get('정책명', 'N/A'),
                    '분야': doc.metadata.get('중분류', 'N/A'),
                    '주관기관': doc.metadata.get('주관기관명', 'N/A'),
                    '신청URL': doc.metadata.get('신청URL', 'N/A'),
                    '신청기간': doc.metadata.get('신청기간', 'N/A')
                })
        
        # 대화 기록 저장
        self._add_to_history("user", question)
        self._add_to_history("assistant", answer)
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_count': len(docs),
            'user_profile': self.get_profile_dict(),
            'question_type': question_type
        }
    
    def search_policies(self, query: str, top_k: int = 5, time_mode: int = None, use_profile: bool = True) -> list:
        """
        정책 검색만 수행 (LLM 쿼리 리라이팅 적용)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수
            time_mode: 시간 모드 (None이면 기본값 사용)
            use_profile: 사용자 프로필 필터링 적용 여부
        """
        if time_mode is None:
            time_mode = self.default_time_mode
        
        # LLM 쿼리 리라이팅
        search_query = rewrite_query_for_search(query, self.llm)
        
        docs = self._search_and_filter(search_query, time_mode, top_k, use_profile)
        
        results = []
        for doc in docs:
            results.append({
                '정책명': doc.metadata.get('정책명', 'N/A'),
                '분야': f"{doc.metadata.get('대분류', '')} > {doc.metadata.get('중분류', '')}",
                '주관기관': doc.metadata.get('주관기관명', 'N/A'),
                '신청기간': doc.metadata.get('신청기간', 'N/A'),
                '지원연령': f"{doc.metadata.get('지원최소연령', '0')}세 ~ {doc.metadata.get('지원최대연령', '0')}세",
                '신청URL': doc.metadata.get('신청URL', 'N/A'),
                '내용': doc.page_content[:200] + '...'
            })
        
        return results
    
    # ===== 프로필 관리 메서드 =====
    
    def set_profile(self, age: int = None, region: str = None, 
                   interests: list = None, employment_status: str = None):
        """
        사용자 프로필 직접 설정
        
        Args:
            age: 나이
            region: 지역
            interests: 관심분야 리스트
            employment_status: 취업상태
        """
        if age is not None:
            self.user_profile.age = age
        if region is not None:
            self.user_profile.region = region
        if interests is not None:
            self.user_profile.interests = interests
        if employment_status is not None:
            self.user_profile.employment_status = employment_status
        
        logger.info(f'✅ 프로필 업데이트: {self.user_profile}')
    
    def get_profile(self) -> str:
        """현재 프로필 문자열로 확인"""
        return str(self.user_profile)
    
    def get_profile_dict(self) -> dict:
        """현재 프로필을 dict로 반환 (Streamlit 대응)"""
        return {
            'age': self.user_profile.age,
            'region': self.user_profile.region,
            'interests': self.user_profile.interests.copy() if self.user_profile.interests else [],
            'employment_status': self.user_profile.employment_status,
            'is_empty': self.user_profile.is_empty()
        }
    
    def clear_profile(self):
        """프로필 초기화"""
        self.user_profile.clear()
        logger.info('✅ 프로필이 초기화되었습니다.')
    
    def show_profile(self):
        """현재 프로필 상세 출력"""
        print('=' * 40)
        print('📋 현재 사용자 프로필')
        print('=' * 40)
        if self.user_profile.is_empty():
            print('설정된 조건이 없습니다.')
        else:
            if self.user_profile.age:
                print(f'  나이: {self.user_profile.age}세')
            if self.user_profile.region:
                print(f'  지역: {self.user_profile.region}')
            if self.user_profile.interests:
                print(f'  관심분야: {", ".join(self.user_profile.interests)}')
            if self.user_profile.employment_status:
                print(f'  취업상태: {self.user_profile.employment_status}')
        print('=' * 40)
    
    # ===== 대화 기록 관리 메서드 =====
    
    def get_chat_history(self) -> list:
        """대화 기록 반환"""
        return self.chat_history.copy()
    
    def clear_chat_history(self):
        """대화 기록 초기화"""
        self.chat_history = []
        logger.info('✅ 대화 기록이 초기화되었습니다.')
    
    def clear_all(self):
        """프로필 + 대화 기록 모두 초기화"""
        self.user_profile.clear()
        self.chat_history = []
        logger.info('✅ 프로필과 대화 기록이 모두 초기화되었습니다.')


logger.info('✅ YouthPolicyRAG 클래스 정의 완료')
logger.debug('   - 하이브리드 전략: 인사(1)/무관(2)/정책(3) 분류 후 처리')
logger.debug('   - LLM 쿼리 리라이팅: rewrite_query_for_search()')
logger.debug('   - 시간 필터링 + 사용자 프로필 필터링 통합')
logger.debug('   - 프로필 관리: set_profile(), get_profile(), clear_profile()')
logger.debug('   - 대화 기록: get_chat_history(), clear_chat_history(), clear_all()')

# %% [markdown]
# ## 5. 테스트 실행
# 
# RAG 시스템을 테스트합니다. (사용자 프로필 필터링 포함)

# %%
# RAG 시스템 초기화
rag = YouthPolicyRAG(db_path=db_path, retriever_k=5)

# %%
# 간단한 검색 테스트 (LLM 호출 없이)
search_results = rag.search_policies('취업 지원', top_k=3)

print('🔍 검색 결과 (취업 지원):\n')
for i, policy in enumerate(search_results, 1):
    print(f"[{i}] {policy['정책명']}")
    print(f"    분야: {policy['분야']}")
    print(f"    주관기관: {policy['주관기관']}")
    print()

# %%
# 질문-답변 테스트 1
question = '취업 지원 정책이 있나요?'

print(f'👤 질문: {question}')
print('─' * 60)

result = rag.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']})")

# %%
# 질문-답변 테스트 2
question = '창업을 준비하는 청년을 위한 정책을 알려주세요'

print(f'👤 질문: {question}')
print('─' * 60)

result = rag.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']})")

# %%
# 질문-답변 테스트 3
question = '주거 지원 받을 수 있는 정책이 있을까요?'

print(f'👤 질문: {question}')
print('─' * 60)

result = rag.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']})")

# %% [markdown]
# ### 5-0. 하이브리드 전략 테스트
# 
# 인사, 무관 질문, 간접 표현 등 다양한 유형의 질문에 대한 응답을 테스트합니다.

# %%
# 하이브리드 전략 테스트용 RAG 초기화
rag_hybrid = YouthPolicyRAG(db_path=db_path, retriever_k=5)

# %%
# 테스트 1: 인사 (유형 1)
test_messages = [
    "안녕",
    "안녕하세요!",
    "하이~"
]

print('=' * 60)
print('🧪 테스트 1: 인사 (유형 1 - GREETING)')
print('=' * 60)

for msg in test_messages:
    print(f'\n👤 메시지: "{msg}"')
    result = rag_hybrid.ask_with_sources(msg)
    print(f'📌 질문 유형: {result["question_type"]}')
    print(f'🤖 답변: {result["answer"][:150]}...' if len(result["answer"]) > 150 else f'🤖 답변: {result["answer"]}')
    print('─' * 40)

# %%
# 테스트 2: 무관 질문 (유형 2)
test_messages = [
    "오늘 날씨 어때?",
    "맛집 추천해줘",
    "BTS 좋아해?"
]

print('=' * 60)
print('🧪 테스트 2: 무관 질문 (유형 2 - IRRELEVANT)')
print('=' * 60)

for msg in test_messages:
    print(f'\n👤 메시지: "{msg}"')
    result = rag_hybrid.ask_with_sources(msg)
    print(f'📌 질문 유형: {result["question_type"]}')
    print(f'🤖 답변: {result["answer"][:150]}...' if len(result["answer"]) > 150 else f'🤖 답변: {result["answer"]}')
    print('─' * 40)

# %%
# 테스트 3: 간접 표현 → 정책 연결 (유형 3 - 추론적 의도 분석)
test_messages = [
    "배고프다",
    "돈이 없어",
    "취업이 안 돼",
    "월세가 너무 비싸",
    "우울해"
]

print('=' * 60)
print('🧪 테스트 3: 간접 표현 → 정책 연결 (유형 3 - 추론적 의도 분석)')
print('=' * 60)

for msg in test_messages:
    print(f'\n👤 메시지: "{msg}"')
    result = rag_hybrid.ask_with_sources(msg)
    print(f'📌 질문 유형: {result["question_type"]}')
    print(f'📚 검색된 정책: {result["retrieved_count"]}개')
    if result["sources"]:
        print(f'   → {result["sources"][0]["정책명"]}')
    print(f'🤖 답변 미리보기: {result["answer"][:200]}...')
    print('─' * 40)

# %%
# 테스트 4: 직접적 정책 질문 (유형 3)
test_messages = [
    "취업 지원 정책 알려줘",
    "주거 지원 받고 싶어",
    "창업 하고 싶은데 지원받을 수 있어?"
]

print('=' * 60)
print('🧪 테스트 4: 직접적 정책 질문 (유형 3 - POLICY)')
print('=' * 60)

for msg in test_messages:
    print(f'\n👤 메시지: "{msg}"')
    result = rag_hybrid.ask_with_sources(msg)
    print(f'📌 질문 유형: {result["question_type"]}')
    print(f'📚 검색된 정책: {result["retrieved_count"]}개')
    for i, src in enumerate(result["sources"][:3], 1):
        print(f'   {i}. {src["정책명"]} ({src["분야"]})')
    print('─' * 40)

# 대화 기록 초기화
rag_hybrid.clear_all()
print('\n✅ 테스트 완료 - 대화 기록 초기화됨')

# %% [markdown]
# ### 5-1. 사용자 프로필 기반 필터링 테스트
# 
# 자연어로 조건을 말하면 자동으로 프로필이 업데이트되고 필터링에 적용됩니다.

# %%
# 프로필 필터링 테스트용 RAG 초기화 (새 세션)
rag_profile = YouthPolicyRAG(db_path=db_path, retriever_k=5)

# %%
# 테스트 1: 나이와 지역 조건을 자연어로 입력
question = "나 25살이고 충북에 살아. 취업 지원 정책 있어?"

print(f'👤 질문: {question}')
print('─' * 60)

result = rag_profile.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📋 현재 프로필: {result["user_profile"]}')
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']}) - {source['주관기관']}")

# %%
# 테스트 2: 이전 조건이 유지되는지 확인 (나이/지역 언급 없이 질문)
question = "주거 지원도 알려줘"

print(f'👤 질문: {question}')
print('─' * 60)

result = rag_profile.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📋 현재 프로필: {result["user_profile"]}')  # 이전 조건이 유지되어야 함
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']}) - {source['주관기관']}")

# %%
# 테스트 3: 조건 변경 (새 지역으로 업데이트)
question = "이제 서울로 이사했어. 창업 지원 뭐 있어?"

print(f'👤 질문: {question}')
print('─' * 60)

result = rag_profile.ask_with_sources(question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📋 현재 프로필: {result["user_profile"]}')  # 지역이 서울로 변경되어야 함
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']} ({source['분야']}) - {source['주관기관']}")

# %%
# 프로필 및 대화 기록 관리 테스트
print('=' * 60)
print('📋 프로필 및 대화 기록 관리 기능 테스트')
print('=' * 60)

# 현재 프로필 확인 (dict 형태)
print('\n▶ 현재 프로필 (dict):')
profile_dict = rag_profile.get_profile_dict()
print(f'  {profile_dict}')

# 현재 프로필 확인 (출력용)
rag_profile.show_profile()

# 대화 기록 확인
print('\n▶ 대화 기록:')
history = rag_profile.get_chat_history()
print(f'  총 {len(history)}개 메시지')
for i, msg in enumerate(history[-4:], 1):  # 최근 4개만 표시
    content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
    print(f'  [{msg["role"]}] {content_preview}')

# 프로필 직접 설정
print('\n▶ 프로필 직접 설정: 30세, 인천, 일자리 관심')
rag_profile.set_profile(age=30, region='인천', interests=['일자리'])
rag_profile.show_profile()

# 전체 초기화
print('\n▶ 전체 초기화 (프로필 + 대화기록)')
rag_profile.clear_all()
rag_profile.show_profile()
print(f'  대화 기록: {len(rag_profile.get_chat_history())}개')

# %% [markdown]
# ## 6. 대화형 챗봇
# 
# 자유롭게 질문을 입력하여 테스트할 수 있습니다.

# %%
# 대화형 챗봇 함수 (프로필 필터링 + 대화 기록 지원)
def interactive_chat(rag_system):
    """대화형 챗봇 실행 (사용자 프로필 기반 필터링 + 대화 기록 지원)"""
    print('\n' + '=' * 60)
    print('💬 청년 정책 RAG 챗봇 (맞춤 필터링 지원)')
    print('=' * 60)
    print('청년 정책에 대해 질문해주세요!')
    print('💡 "나 25살이야", "서울 살아" 등 조건을 말하면 맞춤 검색됩니다.')
    print('💡 명령어:')
    print('   "프로필" - 현재 조건 확인')
    print('   "기록" - 대화 기록 확인')
    print('   "초기화" - 프로필만 리셋')
    print('   "전체초기화" - 프로필 + 대화기록 리셋')
    print('💡 종료: "quit", "q", "종료"\n')
    
    while True:
        try:
            question = input('\n👤 질문: ').strip()
            
            # 종료 명령
            if question.lower() in ['quit', 'q', 'exit', '종료']:
                print('\n👋 챗봇을 종료합니다. 감사합니다!')
                break
            
            # 프로필 확인 명령
            if question in ['프로필', '조건', '내 정보', '내정보']:
                rag_system.show_profile()
                continue
            
            # 대화 기록 확인 명령
            if question in ['기록', '대화기록', '히스토리', 'history']:
                history = rag_system.get_chat_history()
                print(f'\n📜 대화 기록 ({len(history)}개):')
                for i, msg in enumerate(history[-10:], 1):  # 최근 10개만
                    role_icon = "👤" if msg["role"] == "user" else "🤖"
                    content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                    print(f'  {role_icon} {content_preview}')
                continue
            
            # 프로필 초기화 명령
            if question in ['초기화', '리셋', 'reset', 'clear']:
                rag_system.clear_profile()
                continue
            
            # 전체 초기화 명령
            if question in ['전체초기화', '전체리셋', 'clearall', 'reset all']:
                rag_system.clear_all()
                continue
            
            if not question:
                continue
            
            print('\n🔍 관련 정책을 검색하고 답변을 생성 중...')
            
            # 답변 생성
            result = rag_system.ask_with_sources(question)
            
            print(f'\n🤖 답변:\n{result["answer"]}')
            
            # 프로필 정보 표시
            if not result['user_profile'].get('is_empty', True):
                profile = result['user_profile']
                profile_parts = []
                if profile.get('age'): profile_parts.append(f"나이: {profile['age']}세")
                if profile.get('region'): profile_parts.append(f"지역: {profile['region']}")
                if profile.get('interests'): profile_parts.append(f"관심: {', '.join(profile['interests'])}")
                if profile.get('employment_status'): profile_parts.append(f"취업상태: {profile['employment_status']}")
                print(f'\n📋 적용된 필터: {" | ".join(profile_parts)}')
            
            print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source['정책명']} ({source['분야']}) - {source['주관기관']}")
            
        except KeyboardInterrupt:
            print('\n\n👋 챗봇을 종료합니다.')
            break
        except Exception as e:
            logger.error(f'❌ 오류 발생: {e}')

logger.info('✅ 대화형 챗봇 함수 정의 완료')
logger.debug('   - 자연어로 조건 설정: "나 25살이야", "인천 살아" 등')
logger.debug('   - 대화 기록 자동 저장 (최대 20개)')
logger.debug('   - 명령어: "프로필", "기록", "초기화", "전체초기화"')
print('아래 셀을 실행하면 대화형 모드가 시작됩니다.')

# %%
# 대화형 챗봇 실행 (필요시 주석 해제)
# interactive_chat(rag)

# %% [markdown]
# ## 7. 직접 질문 테스트
# 
# 아래 셀에서 직접 질문을 입력하여 테스트해보세요.

# %%
# 직접 질문 입력 (원하는 질문으로 수정하세요)
my_question = '취업 지원 정책 알려줘'

print(f'👤 질문: {my_question}')
print('─' * 60)

result = rag.ask_with_sources(my_question)

print(f'\n🤖 답변:\n{result["answer"]}')
print(f'\n📚 참고 정책 ({result["retrieved_count"]}개):')
for i, source in enumerate(result['sources'], 1):
    print(f"  {i}. {source['정책명']}")
    print(f"      분야: {source['분야']}")
    print(f"      신청기간: {source['신청기간']}")
    print(f"      주관기관: {source['주관기관']}")
    if source['신청URL'] != 'N/A' and source['신청URL']:
        print(f"      신청URL: {source['신청URL']}")
    print()


