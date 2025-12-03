import streamlit as st
import os
import logging
from datetime import datetime
import json
import re
from typing import Optional, List
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# 1. RAG 시스템에서 필요한 핵심 클래스 및 함수 재정의
#    (youth_policy_rag.ipynb 파일의 코드를 기반으로 재정의)
# ----------------------------------------------------------------------

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('streamlit_rag_test')

# ** UserProfile 클래스 **
@dataclass
class UserProfile:
    age: Optional[int] = None
    region: Optional[str] = None
    interests: List[str] = field(default_factory=list)
    employment_status: Optional[str] = None
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                if key == 'interests' and isinstance(value, str):
                    if value not in self.interests:
                        self.interests.append(value)
                elif key == 'interests' and isinstance(value, list):
                    self.interests = value
                else:
                    setattr(self, key, value)
    
    def clear(self):
        self.age = None
        self.region = None
        self.interests = []
        self.employment_status = None
    
    def is_empty(self) -> bool:
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

# ** 날짜 파싱 및 필터링 함수 **
def parse_date_range(date_str: str) -> tuple:
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return (None, None)
    try:
        parts = re.split(r'\s*[~\-]\s*', date_str.strip())
        if len(parts) == 2:
            start_str, end_str = parts
            start_date = datetime.strptime(start_str.strip(), '%Y%m%d')
            end_date = datetime.strptime(end_str.strip(), '%Y%m%d')
            return (start_date, end_date)
        elif len(parts) == 1:
            single_date = datetime.strptime(parts[0].strip(), '%Y%m%d')
            return (single_date, single_date)
    except (ValueError, AttributeError):
        pass
    return (None, None)

def is_currently_available(date_str: str, today: datetime) -> bool:
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return True
    start_date, end_date = parse_date_range(date_str)
    if start_date is None or end_date is None:
        return True
    return start_date <= today <= end_date

def is_this_year(date_str: str, year: int) -> bool:
    if not date_str or date_str == 'N/A' or date_str.strip() == '':
        return True
    start_date, end_date = parse_date_range(date_str)
    if start_date is None or end_date is None:
        return True
    year_start = datetime(year, 1, 1)
    year_end = datetime(year, 12, 31)
    return not (end_date < year_start or start_date > year_end)

def filter_docs_by_time(docs: list, time_mode: int, today: datetime = None) -> list:
    if today is None:
        today = datetime.now()
    if time_mode == 3:
        return docs
    
    filtered = []
    current_year = today.year

    for doc in docs:
        date_str = doc.metadata.get('신청기간', '')
        if time_mode == 1:
            if is_currently_available(date_str, today):
                filtered.append(doc)
        elif time_mode == 2:
            if is_this_year(date_str, current_year):
                filtered.append(doc)
    
    return filtered

# ** 프로필 기반 필터링 함수 **
def is_age_eligible(doc, user_age: int) -> bool:
    if user_age is None:
        return True
    min_age = doc.metadata.get('지원최소연령', 0)
    max_age = doc.metadata.get('지원최대연령', 0)
    try:
        min_age = int(min_age) if min_age else 0
        max_age = int(max_age) if max_age else 0
    except (ValueError, TypeError):
        min_age, max_age = 0, 0
    if min_age == 0 and max_age == 0:
        return True
    if min_age == 0:
        return user_age <= max_age
    if max_age == 0:
        return user_age >= min_age
    return min_age <= user_age <= max_age

def is_region_match(doc, user_region: str) -> bool:
    if user_region is None:
        return True
    org_name = doc.metadata.get('주관기관명', '') + doc.metadata.get('등록기관명', '')
    region_keywords = {
        '서울': ['서울'], '부산': ['부산'], '대구': ['대구'], '인천': ['인천'],
        '광주': ['광주'], '대전': ['대전'], '울산': ['울산'], '세종': ['세종'],
        '경기': ['경기'], '강원': ['강원'],
        '충북': ['충북', '충청북도'], '충남': ['충남', '충청남도'],
        '전북': ['전북', '전라북도'], '전남': ['전남', '전라남도'],
        '경북': ['경북', '경상북도'], '경남': ['경남', '경상남도'],
        '제주': ['제주'],
    }
    national_keywords = ['고용노동부', '보건복지부', '국토교통부', '중소벤처기업부', 
                        '교육부', '문화체육관광부', '여성가족부', '행정안전부', '지방자치단체', '전국']
    for keyword in national_keywords:
        if keyword in org_name:
            return True
    keywords = region_keywords.get(user_region, [user_region])
    for keyword in keywords:
        if keyword in org_name:
            return True
    return False

def is_interest_match(doc, interests: list) -> bool:
    if not interests:
        return True
    policy_category = doc.metadata.get('대분류', '') + doc.metadata.get('중분류', '')
    for interest in interests:
        if interest in policy_category:
            return True
    return False

def is_employment_match(doc, employment_status: str) -> bool:
    if employment_status is None:
        return True
    exclude_target = doc.metadata.get('참여제외대상', '')
    requirements = doc.metadata.get('추가자격조건', '')
    policy_content = doc.page_content
    if employment_status == '미취업':
        if ('재직' in requirements and '미취업' not in requirements) or '재직' in exclude_target:
            return False
        return True
    elif employment_status == '재직중':
        if '미취업' in requirements and '재직' not in requirements:
            return False
        return True
    elif employment_status == '창업':
        if '창업' in policy_content or '사업자' in policy_content or '창업' in requirements:
            return True
        return True 
    elif employment_status == '학생':
        if '대학생' in exclude_target or '재학생' in exclude_target:
            return False
        return True
    return True

def filter_docs_by_profile(docs: list, profile: 'UserProfile') -> list:
    if profile is None or profile.is_empty():
        return docs
    filtered = []
    for doc in docs:
        if (is_age_eligible(doc, profile.age) and
            is_region_match(doc, profile.region) and
            is_interest_match(doc, profile.interests) and
            is_employment_match(doc, profile.employment_status)):
            filtered.append(doc)
    return filtered

# ** LangChain 및 RAG 구성 요소 **

try:
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document

    # ========================================
    # 질문 유형 분류기 (하이브리드 전략)
    # ========================================
    QUESTION_TYPE_CLASSIFIER_PROMPT = '''당신은 청년 정책 챗봇의 질문 분류기입니다.
사용자의 메시지를 분석하여 다음 3가지 유형 중 하나로 분류하세요.

## 분류 기준
1: 인사/안부 (RAG 불필요)
   - "안녕", "반가워", "고마워", "잘 가", "좋은 하루" 등
   - 단순한 인사말이나 감사 표현

2: 정책과 무관한 질문 (RAG 불필요)
   - "오늘 날씨 어때?", "맛집 추천해줘", "농담 해줘" 등
   - 청년 정책과 전혀 관련 없는 일반 질문

3: 정책 관련 질문 (RAG 필요)
   - 직접적: "취업 지원 정책 알려줘", "주거 보조금 뭐 있어?"
   - 간접적: "배고파", "돈이 없어", "집이 필요해", "일자리 구하기 힘들어"
   → 간접적 표현도 생계/주거/취업 등 정책으로 연결 가능하면 3으로 분류!

## 중요 규칙
- "배고프다", "돈이 없어", "생활이 힘들어" → 생계/복지 정책 관련 (3으로 분류)
- "집이 없어", "월세가 비싸" → 주거 정책 관련 (3으로 분류)
- "일자리가 없어", "취업이 안 돼" → 일자리 정책 관련 (3으로 분류)
- 애매하면 3으로 분류 (정책 검색 시도)

숫자만 응답하세요 (1, 2, 또는 3)

메시지: {message}

분류:'''

    def classify_question_type(message: str, llm) -> int:
        """질문 유형 분류: 1(인사), 2(무관), 3(정책)"""
        prompt = ChatPromptTemplate.from_template(QUESTION_TYPE_CLASSIFIER_PROMPT)
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({'message': message})
            q_type = int(result.strip()[0])
            if q_type in [1, 2, 3]:
                logger.info(f"질문 유형 분류: {q_type}")
                return q_type
        except (ValueError, IndexError) as e:
            logger.warning(f"질문 유형 분류 실패: {e}")
        return 3  # 기본값: 정책 관련으로 간주

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
   - "추워", "난방비" → "에너지 난방 복지 지원"

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
        """LLM을 사용하여 사용자 메시지를 정책 검색에 적합한 쿼리로 변환"""
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        try:
            rewritten = chain.invoke({'message': message}).strip()
            logger.info(f"쿼리 리라이팅: '{message}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"쿼리 리라이팅 실패: {e}, 원본 사용")
            return message

    # 인사 응답용 프롬프트
    GREETING_PROMPT = '''당신은 "청년이음"의 스마트한 선배입니다.
사용자가 인사를 했습니다. 친근하게 인사로 응답하고, 청년 정책에 대해 도움을 줄 수 있다고 안내하세요.

## 규칙
- 반말로 친근하게 응답
- 짧고 밝게 인사
- 청년 정책 관련 질문을 유도

사용자 메시지: {message}

응답:'''

    # 무관한 질문 응답용 프롬프트
    IRRELEVANT_PROMPT = '''당신은 "청년이음"의 스마트한 선배입니다.
사용자가 청년 정책과 관련 없는 질문을 했습니다. 정중하게 안내하세요.

## 규칙
- 반말로 친근하게 응답
- 정책 관련 질문을 할 수 있도록 유도
- 예시 질문 제안

사용자 메시지: {message}

응답:'''

    # LLM 기반 의도 분류기 
    INTENT_CLASSIFIER_PROMPT = '''당신은 청년 정책 질문의 의도를 분류하는 분류기입니다.
사용자의 질문을 분석하여 어떤 시간 범위의 정책을 원하는지 판단하세요.

## 분류 기준
1: 현재 신청 가능한 정책 (기본값)
   - 일반적인 정책 질문 ("취업 지원 뭐 있어?", "주거 정책 알려줘")
   - "지금", "현재", "신청 가능한" 등의 키워드
   
2: 올해 진행 중인 정책
   - "올해", "이번 년도", "2025년", "금년" 등의 키워드
   
3: 시간 무관 (역대 전체)
   - "전체", "모든", "역대", "지금까지", "과거", "종료된 것도"

## 규칙
- 명확한 시간 키워드가 없으면 1(현재 신청 가능)로 분류
- 숫자만 응답하세요 (1, 2, 또는 3)

질문: {question}

분류 (1/2/3):'''

    def classify_time_intent(question: str, llm) -> int:
        prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFIER_PROMPT)
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({'question': question})
            time_mode = int(result.strip()[0])
            if time_mode in [1, 2, 3]:
                return time_mode
        except (ValueError, IndexError):
            pass
        return 1

    # LLM 기반 사용자 조건 추출기
    PROFILE_EXTRACTOR_PROMPT = '''당신은 사용자의 대화에서 개인 조건 정보를 추출하는 분석기입니다.
사용자의 메시지를 분석하여 다음 정보를 JSON 형식으로 추출하세요.

## 추출 항목
1. age (정수 또는 null): 나이
2. region (문자열 또는 null): 지역 (시/도 단위로 정규화, 예: 서울, 경기, 충북 등)
3. interest (문자열 또는 null): 관심 분야 (대분류, 예: 일자리, 주거, 교육, 복지문화, 참여권리)
4. employment_status (문자열 또는 null): 취업 상태 (예: 미취업, 재직중, 창업, 학생)

## 규칙
- 명확하게 언급된 정보만 추출
- 추측하지 말 것
- 반드시 JSON 형식으로만 응답하며, 코드 블록(```json)으로 감싸지 말 것.

사용자 메시지: {message}

JSON 출력:'''

    def extract_user_profile(message: str, llm) -> dict:
        prompt = ChatPromptTemplate.from_template(PROFILE_EXTRACTOR_PROMPT)
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({'message': message})
            result = result.strip()
            # 혹시 LLM이 ```json ... ``` 형태로 응답하면 정리
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

    # RAG 시스템 클래스
    class YouthPolicyRAG:
        SYSTEM_PROMPT = '''당신은 \"청년이음\"의 스마트한 선배입니다.
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
- "집이 없어", "월세가 너무 비싸", "잘곳이 없어" → 주거 지원 정책 추천
- "취업이 안 돼", "일자리가 없어" → 일자리/취업 지원 정책 추천
- "학비가 부담돼", "공부하고 싶어" → 교육/장학금 정책 추천
- "우울해", "지쳐", "스트레스 받아" → 청년 마음건강/복지 정책 추천
- "추워", "난방비가 부담돼" → 에너지/난방 지원 정책 추천
- "창업하고 싶어", "사업 아이디어가 있어" → 창업 지원 정책 추천

## 답변 형식 (반드시 이 구조를 따르세요)

### 1️⃣ 인사/공감 (첫 1~2문장)
밝고 긍정적인 분위기로 시작하세요.
예시:
- "힘든 상황이구나... 도움 될 만한 정책 찾아봤어!"
- "요즘 이 정책 궁금해하는 사람 많더라구!"

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

### 5️⃣ 추가 안내
마지막에 항상 이 문장을 포함하세요:
"더 궁금한 거 있으면 편하게 물어봐! 😊"

## 정책 미검색 시 안내
검색된 정책이 없거나 적합한 정책을 찾지 못한 경우:
"음, 딱 맞는 정책을 찾기가 좀 어렵네! 혹시 이런 분야에 관심 있어?
- 🏢 일자리: 취업 지원, 인턴, 창업
- 🏠 주거: 전월세, 청년 주택
- 🎓 교육: 장학금, 학자금
- 🎨 복지: 청년 수당, 마음 건강
좀 더 구체적으로 알려주면 더 잘 찾아볼게! 😊"

## 답변 규칙
1. **검색된 정책이 있으면 반드시 해당 정책을 안내하세요!** (가장 중요)
2. 제공된 정책 정보만을 기반으로 답변하세요.
3. 어려운 행정 용어는 쉽게 풀어서 설명하세요.
4. 여러 정책이 있으면 각각 위 형식으로 안내하세요.
5. 반말을 사용하되 존중하는 톤을 유지하세요.
6. 신청기간은 제공된 원본 데이터를 그대로 표시하고, "N/A"이거나 비어있으면 "상시 모집 또는 별도 공지"라고 안내하세요.
'''
        HUMAN_PROMPT = '''
[검색된 관련 정책 정보]:
{context}

질문: {question}

답변:'''

        def __init__(self, db_path, retriever_k=5, search_k_multiplier=4, default_time_mode=1):
            self.db_path = db_path
            self.retriever_k = retriever_k
            self.search_k = retriever_k * search_k_multiplier
            self.default_time_mode = default_time_mode
            
            # 1. LLM 및 임베딩 모델 로드
            self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
            self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
            
            # 2. VectorDB 로드
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.db_path,
                    collection_name='youth_policies',
                    embedding_function=self.embedding_model
                )
                self.total_policies = self.vectorstore._collection.count()
                st.session_state.rag_status = f"✅ VectorDB 로드 완료 (총 {self.total_policies}개 정책)"
            except Exception as e:
                st.session_state.rag_status = f"❌ VectorDB 로드 실패: {e}"
                raise e

            # 3. 프롬프트 템플릿
            self.prompt = ChatPromptTemplate.from_messages([
                ('system', self.SYSTEM_PROMPT),
                ('human', self.HUMAN_PROMPT)
            ])
            
            st.session_state.rag_status += f" | LLM 및 Retriever 설정 완료 (K={self.retriever_k})"
            st.session_state.rag_system_ready = True
    
        def _classify_intent(self, question: str) -> int:
            return classify_time_intent(question, self.llm)
        
        def _handle_greeting(self, message: str) -> str:
            """인사 메시지 처리 (RAG 없이 LLM만 사용)"""
            prompt = ChatPromptTemplate.from_template(GREETING_PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({'message': message})
        
        def _handle_irrelevant(self, message: str) -> str:
            """무관한 질문 처리 (RAG 없이 LLM만 사용)"""
            prompt = ChatPromptTemplate.from_template(IRRELEVANT_PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({'message': message})
        
        def _extract_and_update_profile(self, message: str):
            extracted = extract_user_profile(message, self.llm)
            
            current_profile = st.session_state.user_profile
            
            # 프로필 업데이트 (None이 아닌 값만)
            if extracted.get('age') is not None:
                current_profile.age = extracted['age']
            if extracted.get('region') is not None:
                current_profile.region = extracted['region']
            if extracted.get('interest') is not None:
                interest = extracted['interest']
                if interest not in current_profile.interests:
                    current_profile.interests.append(interest)
            if extracted.get('employment_status') is not None:
                current_profile.employment_status = extracted['employment_status']
            
            st.session_state.user_profile = current_profile
        
        def _search_and_filter(self, query: str, time_mode: int, top_k: int) -> list:
            # 1. 시맨틱 검색 (더 넓게 검색)
            docs = self.vectorstore.similarity_search(query, k=self.search_k)
            
            # 2. 시간 기반 필터링
            filtered_docs = filter_docs_by_time(docs, time_mode)
            
            # 3. 사용자 프로필 기반 필터링
            if not st.session_state.user_profile.is_empty():
                filtered_docs = filter_docs_by_profile(filtered_docs, st.session_state.user_profile)
            
            # 최종 top_k 반환
            return filtered_docs[:top_k]

        def _format_docs(self, docs: list) -> str:
            formatted = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                doc_content = doc.page_content if isinstance(doc, Document) else doc.get('page_content', '')
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
{doc_content}
"""
                formatted.append(text)
            return '\n\n---\n\n'.join(formatted)

        def ask_with_sources(self, question: str, time_mode: int) -> dict:
            if not st.session_state.rag_system_ready:
                 return {'answer': "❌ RAG 시스템이 초기화되지 않았습니다. 오류 메시지를 확인하세요.", 'sources': [], 'retrieved_count': 0, 'time_mode_used': 0, 'question_type': 0}

            # ========================================
            # 하이브리드 전략: 질문 유형 분류 먼저!
            # ========================================
            question_type = classify_question_type(question, self.llm)
            
            # 1. 인사 메시지 처리 (RAG 불필요)
            if question_type == 1:
                answer = self._handle_greeting(question)
                return {
                    'answer': answer,
                    'sources': [],
                    'retrieved_count': 0,
                    'time_mode_used': 0,
                    'question_type': 1
                }
            
            # 2. 무관한 질문 처리 (RAG 불필요)
            if question_type == 2:
                answer = self._handle_irrelevant(question)
                return {
                    'answer': answer,
                    'sources': [],
                    'retrieved_count': 0,
                    'time_mode_used': 0,
                    'question_type': 2
                }
            
            # ========================================
            # 3. 정책 관련 질문 처리 (RAG 사용)
            # ========================================
            
            # LLM 쿼리 리라이팅 (간접 표현 → 정책 키워드)
            search_query = rewrite_query_for_search(question, self.llm)
            
            # 사용자 프로필 추출 및 업데이트
            self._extract_and_update_profile(question)
            
            # 시간 모드 결정
            if time_mode == 0:
                time_mode = self._classify_intent(question)
            
            # 검색 및 필터링 (리라이팅된 검색어 사용)
            docs = self._search_and_filter(search_query, time_mode, self.retriever_k)
            
            # 컨텍스트 생성
            context = self._format_docs(docs)
            
            # LLM 답변 생성
            today_str = datetime.now().strftime('%Y-%m-%d')
            chain = self.prompt | self.llm | StrOutputParser()
            
            # 정책이 없는 경우 예외 처리 및 정책 무관 질문 처리
            if not context:
                answer = chain.invoke({
                    'context': '',
                    'question': '단순 인사 또는 정책과 무관한 질문일 경우 규칙 4에 따라 답변해 주세요. (질문 자체: ' + question + ')',
                    'today': today_str
                })
            else:
                answer = chain.invoke({
                    'context': context,
                    'question': question,
                    'today': today_str
                })
            
            # 출처 정보 추출
            sources = []
            for doc in docs:
                sources.append({
                    '정책명': doc.metadata.get('정책명', 'N/A'),
                    '분야': doc.metadata.get('중분류', 'N/A'),
                    '주관기관': doc.metadata.get('주관기관명', 'N/A'),
                    '신청URL': doc.metadata.get('신청URL', 'N/A'),
                    '신청기간': doc.metadata.get('신청기간', 'N/A')
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'retrieved_count': len(docs),
                'time_mode_used': time_mode,
                'question_type': 3
            }

except ImportError as e:
    st.error(f"❌ 라이브러리 로드 오류: LangChain, OpenAI, Chroma 등의 라이브러리가 설치되지 않았거나 설정되지 않았습니다. `pip install langchain_openai chromadb python-dotenv`를 실행하세요. 오류: {e}")
    st.stop()
except Exception as e:
    # 환경변수 오류일 가능성이 높음
    st.error(f"❌ 초기 설정 중 치명적인 오류 발생: {e}")
    st.stop()


# ----------------------------------------------------------------------
# 2. Streamlit UI 구성 및 초기화
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="청년 정책 RAG 테스트 챗봇 (대화형)",
    layout="wide"
)

st.title("💬 청년 정책 RAG 테스트 챗봇")
st.caption("대화 기록이 누적되며, 프로필 정보가 자동으로 추출/적용됩니다.")

# --- 세션 상태 초기화 (오류 방지) ---
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = UserProfile()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_system_ready' not in st.session_state:
    st.session_state.rag_system_ready = False
if 'rag_status' not in st.session_state:
    st.session_state.rag_status = "RAG 시스템 초기화 대기 중..."

# --- RAG 시스템 초기화 (VectorDB 경로 설정) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if current_dir.endswith('src') else current_dir
db_path = os.path.join(project_root, 'data', 'vectordb')

def initialize_rag():
    try:
        if 'rag' not in st.session_state or not st.session_state.rag_system_ready:
             st.session_state.rag = YouthPolicyRAG(db_path=db_path)
             st.success(st.session_state.rag_status)
    except Exception as e:
        st.error(f"RAG 시스템 초기화 실패: {e}")

# --------------------
# A. 사이드바: 프로필 설정 및 상태
# --------------------

with st.sidebar:
    st.header("🛠️ 테스트 설정 및 프로필")
    
    # 1. 시스템 상태 및 초기화 버튼
    st.subheader("시스템 상태")
    st.markdown(f"**상태:** {st.session_state.rag_status}")
    st.info(f"VectorDB 경로: `{db_path}`")
    
    if st.button("🚀 RAG 시스템 초기화 시작 (VectorDB 로드)"):
        initialize_rag()
    
    if 'rag' in st.session_state and st.session_state.rag_system_ready:
        st.markdown(f"**총 정책 수:** {st.session_state.rag.total_policies}개")
        st.markdown(f"**최종 반환 K:** {st.session_state.rag.retriever_k}개")
    
    # 2. 프로필 수동 설정 (테스트용)
    st.subheader("사용자 프로필 직접 설정")
    profile = st.session_state.user_profile
    
    age = st.number_input("나이 (세)", min_value=0, max_value=100, value=profile.age if profile.age else 0, format="%d")
    region_options = ['', '서울', '부산', '대구', '인천', '광주', '대전', '울산', '세종', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
    region = st.selectbox("지역 (시/도)", region_options, index=region_options.index(profile.region) if profile.region in region_options else 0)
    
    all_interests = ['일자리', '주거', '교육', '복지문화', '참여권리']
    interests = st.multiselect("관심 분야", all_interests, default=profile.interests)

    employment_status_options = ['', '미취업', '재직중', '창업', '학생']
    employment_status = st.selectbox("취업 상태", employment_status_options, index=employment_status_options.index(profile.employment_status) if profile.employment_status in employment_status_options else 0)

    if st.button("프로필 수동 업데이트"):
        new_age = age if age > 0 else None
        new_region = region if region else None
        new_interests = interests if interests else []
        new_employment_status = employment_status if employment_status else None
        
        st.session_state.user_profile.age = new_age
        st.session_state.user_profile.region = new_region
        st.session_state.user_profile.interests = new_interests
        st.session_state.user_profile.employment_status = new_employment_status
        st.success("✅ 프로필이 수동으로 업데이트되었습니다.")

    col_clear_1, col_clear_2 = st.columns(2)
    with col_clear_1:
        if st.button("👤 프로필 초기화", help="모든 필터링 조건을 초기화합니다."):
            st.session_state.user_profile.clear()
            st.success("✅ 프로필 조건이 초기화되었습니다.")
    with col_clear_2:
        if st.button("🗑️ 대화 기록 삭제", help="화면의 모든 대화 기록을 삭제합니다."):
            st.session_state.messages = []
            st.success("✅ 대화 기록이 삭제되었습니다.")
            st.rerun()

    # 3. 현재 적용된 프로필 (상태 확인)
    st.subheader("현재 적용된 조건")
    st.code(str(st.session_state.user_profile), language='text')

# --------------------
# C. 메인 화면: 채팅 인터페이스
# --------------------

if st.session_state.rag_system_ready and 'rag' in st.session_state:
    
    rag_system = st.session_state.rag
    
    # --- 1. 대화 기록 누적 출력 ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # 사용자 메시지 아래에 프로필 업데이트/적용 정보 캡션 출력 (디버그용)
            if message["role"] == "user" and message.get("profile_update_info"):
                 st.caption(message['profile_update_info'])

    # --- 2. 질문 입력 및 답변 생성 ---
    
    # 시간 모드 설정 (자동 판단을 기본값으로 사용)
    time_mode_options = {'자동 판단': 0, '현재 신청 가능 (1)': 1, '올해 진행 중 (2)': 2, '시간 무관 (3)': 3}
    time_mode_key = st.selectbox("시간 필터 모드 선택", list(time_mode_options.keys()), index=1)
    time_mode = time_mode_options[time_mode_key]

    prompt = st.chat_input("정책에 대해 질문해주세요. (예: 25살이고 서울 살아. 주거 지원 뭐 있어?)") 

    if prompt:
        
        # 1. 사용자 질문을 기록 및 화면 출력 (일단 최소한의 정보만 기록)
        st.session_state.messages.append({"role": "user", "content": prompt, "profile_update_info": None})
        
        # Streamlit이 다시 실행되면서 새로운 메시지를 출력할 것이므로, 
        # 여기서는 스피너를 보여주기 위해 사용자 메시지를 한 번 더 출력합니다.
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. RAG 시스템 호출 및 답변 생성
        with st.spinner("🔍 질문 분석 및 답변 생성 중..."):
            try:
                result = rag_system.ask_with_sources(prompt, time_mode)
                
                answer = result["answer"]
                retrieved_count = result["retrieved_count"]
                question_type = result.get("question_type", 3)
                
                # 질문 유형 표시
                type_labels = {1: "인사", 2: "정책 외 질문", 3: "정책 질문"}
                type_label = type_labels.get(question_type, "정책 질문")
                
                # 디버그 정보 (프로필 업데이트)
                profile_str = str(st.session_state.user_profile)
                profile_update_info = f"🏷️ 질문유형: {type_label} | 👤 조건: {profile_str} | ⏱️ 시간: {time_mode_key} | 📊 검색: {retrieved_count}개"

                # 3. 답변 메시지 구성 (LLM 응답 + 출처/디버그 정보)
                full_response = [answer]
                
                # 출처 정보 추가
                if result['sources'] and not answer.startswith("음, 아쉽지만"):
                    source_list = "\n".join([
                        f"- **{s['정책명']}** ({s['분야']} / {s['주관기관']})" 
                        for s in result['sources']
                    ])
                    full_response.append("\n\n---\n\n")
                    full_response.append(f"### 📚 참고 정책 ({retrieved_count}개)")
                    full_response.append(source_list)

                # 4. 챗봇 답변을 기록 및 화면 출력
                st.session_state.messages.append({"role": "assistant", "content": "\n".join(full_response)})
                
                # 직전 사용자 메시지에 프로필 업데이트 정보 캡션 추가 (디버그용)
                st.session_state.messages[-2]["profile_update_info"] = profile_update_info
            
            except Exception as e:
                error_message = f"❌ 답변 생성 중 오류 발생: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                
        # 스크립트 재실행을 요청하여 화면을 갱신하고 새로운 답변을 보여줍니다.
        st.rerun()

else:
    st.warning("RAG 시스템이 초기화되지 않았습니다. 사이드바에서 'RAG 시스템 초기화 시작' 버튼을 눌러주세요.")