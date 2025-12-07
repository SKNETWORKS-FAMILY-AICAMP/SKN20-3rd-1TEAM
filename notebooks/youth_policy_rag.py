"""
청년 정책 RAG Pipeline
Streamlit과 CLI 모두에서 사용 가능한 모듈
"""

import os
from dotenv import load_dotenv
import chromadb
import json
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# Ensure scikit-learn is available for TF-IDF / BM25 retrievers
try:
    import sklearn  # noqa: F401
except Exception as _e:
    raise ImportError("Could not import scikit-learn, please install with pip install scikit-learn.")

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# Streamlit 환경에서는 print 출력을 최소화
def safe_print(message, force=False):
    """Streamlit에서는 print를 억제하고 필요한 경우만 출력"""
    import sys
    # Streamlit 환경 체크
    if force or 'streamlit' not in sys.modules:
        print(message)


class SimpleEnsembleRetriever:
    # 앙상블 기반 검색기를 진짜 만들고 싶었는데 이게 import 가 안되서 직접 구현한 버전으로 쓸수 밖에 없었습니다..
    """3-way Ensemble Retriever 구현 (Dense + BM25 + TF-IDF)"""
    
    def __init__(self, retrievers, weights):
        """
        Args:
            retrievers: List of retrievers [vector, bm25, tfidf]
            weights: List of weights [0.5, 0.3, 0.2]
        """
        self.retrievers = retrievers
        self.weights = weights
    
    def get_relevant_documents(self, query):
        """각 retriever에서 문서를 가져와 가중치 기반으로 결합"""
        all_docs = []
        
        # 각 retriever에서 검색
        for retriever, weight in zip(self.retrievers, self.weights):
            try:
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
                # 가중치 적용 (점수가 있으면 곱하기, 없으면 순위 기반)
                for i, doc in enumerate(docs):
                    # 간단한 점수 부여: (전체 개수 - 순위) * 가중치
                    score = (len(docs) - i) * weight
                    all_docs.append((doc, score))
            except Exception as e:
                safe_print(f"⚠️ Retriever 오류: {e}")
                continue
        
        # 점수 기준 정렬
        all_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 중복 제거
        seen_ids = set()
        unique_docs = []
        for doc, score in all_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs[:10]  # 상위 10개


class MultiQueryGenerator:
    """질문을 여러 관점으로 재작성하는 MultiQuery 생성기"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self._create_prompt()
    
    def _create_prompt(self):
        """MultiQuery 프롬프트 생성"""
        template = """당신은 AI 검색 전문가입니다. 사용자의 질문을 약간 다른 표현으로 재작성하여 더 나은 검색 결과를 얻으려고 합니다.

원본 질문: {question}

**중요**: 원본 질문의 핵심 의도는 절대 변경하지 마세요. 단지 표현만 살짝 바꾸세요.

위 질문을 **2가지 방식**으로만 재작성하세요:
1. 동일한 의미를 다른 단어로 표현 (예: "창업 지원" → "스타트업 지원")
2. 좀 더 구체적인 키워드 추가 (예: "취업 지원" → "청년 취업 지원 프로그램")

응답 형식 (JSON):
{{
  "queries": [
    "재작성된 질문 1",
    "재작성된 질문 2"
  ]
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def generate_queries(self, question):
        """질문을 여러 개로 확장"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            
            # JSON 파싱
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            expanded_queries = result.get("queries", [])
            
            # 원본 질문을 항상 첫 번째로 포함
            queries = [question] + expanded_queries
            
            safe_print(f"🔄 MultiQuery 생성: {len(queries)}개 (원본 포함)")
            for i, q in enumerate(queries, 1):
                safe_print(f"  {i}. {q}")
            
            return queries
            
        except Exception as e:
            safe_print(f"⚠️ MultiQuery 생성 실패: {e}, 원본 질문만 사용")
            return [question]


class YouthPolicyRAG:
    """청년 정책 RAG 시스템"""
    
    def __init__(self, db_path="../data/vectordb", use_multi_query=True, use_multi_agent = False):
        """
        초기화
        
        Args:
            db_path: ChromaDB 경로
            use_multi_query: MultiQuery 사용 여부 (기본: True)
        """
        safe_print("🚀 RAG Pipeline 초기화 중...")
        
        # LLM 초기화
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
                api_key=OPENAI_API_KEY
            )
        except Exception as e:
            safe_print(f"❌ LLM 초기화 오류: {e}", force=True)
            raise
        
        # 임베딩 모델
        try:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=OPENAI_API_KEY
            )
        except Exception as e:
            safe_print(f"❌ 임베딩 모델 초기화 오류: {e}", force=True)
            raise
        
        # Vector Store 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_db_path = os.path.join(current_dir, db_path)
        
        safe_print(f"📁 벡터DB 경로: {full_db_path}", force=True)
        
        try:
            self.vectorstore = Chroma(
                persist_directory=full_db_path,
                collection_name="youth_policies",
                embedding_function=self.embeddings
            )
            safe_print(f"✅ ChromaDB 연결 성공")
        except Exception as e:
            safe_print(f"❌ ChromaDB 연결 오류: {e}", force=True)
            raise
        
        # ChromaDB collection 직接 접근 (필터링용)
        try:
            chroma_client = chromadb.PersistentClient(path=full_db_path)
            self.collection = chroma_client.get_collection(name="youth_policies")
            safe_print(f"✅ Collection 접근 성공")
        except Exception as e:
            safe_print(f"❌ Collection 접근 오류: {e}", force=True)
            raise
        
        # 문서 로딩 (한 번만)
        self.documents = self._load_documents()
        
        # BM25 Retriever 초기화 (키워드 기반 검색)
        self._init_bm25_retriever()
        
        # TF-IDF Retriever 초기화 (통계 기반 검색)
        self._init_tfidf_retriever()
        
        # Ensemble Retriever 생성 (Dense + BM25 + TF-IDF)
        self._init_ensemble_retriever()
        
        # MultiQuery Generator 초기화
        self.multi_query_gen = MultiQueryGenerator(self.llm)
        
        # 사용자 정보 (나이, 지역, 학력)
        self.user_age = None
        self.user_region = None
        self.user_education = None
        
        # MultiQuery 사용 여부
        self.use_multi_query = use_multi_query
        
        # Router 프롬프트
        self.router_prompt = self._create_router_prompt()

        self.chat_history = []      # 대화 메모리용 리스트
        self.self_rag_prompt = self._create_self_rag_prompt()  # Self-RAG 프롬프트
        self.use_multi_agent = use_multi_agent
        
        safe_print("✅ RAG Pipeline 초기화 완료!")
    
    def _load_documents(self):
        """ChromaDB에서 문서 로딩 (한 번만 수행)"""
        safe_print("📄 문서 로딩 중...")
        all_data = self.collection.get()
        
        documents = []
        for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
            documents.append(Document(
                page_content=doc_text,
                metadata=metadata
            ))
        
        safe_print(f"✅ 문서 로딩 완료 (문서 수: {len(documents)}개)")
        return documents
    
    def _init_bm25_retriever(self):
        """BM25 Retriever 초기화 (키워드 기반 검색)"""
        safe_print("📚 BM25 Retriever 초기화 중...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30  # 상위 30개 검색 (다양성 확보)
        safe_print("✅ BM25 Retriever 초기화 완료")
    
    def _init_tfidf_retriever(self):
        """TF-IDF Retriever 초기화 (통계 기반 검색)"""
        safe_print("📊 TF-IDF Retriever 초기화 중...")
        self.tfidf_retriever = TFIDFRetriever.from_documents(self.documents)
        self.tfidf_retriever.k = 30  # 상위 30개 검색 (다양성 확보)
        safe_print("✅ TF-IDF Retriever 초기화 완료")
    
    def _init_ensemble_retriever(self):
        """Ensemble Retriever 초기화 (Dense + BM25 + TF-IDF 3-way hybrid)"""
        safe_print("🔗 Ensemble Retriever 생성 중 (3-way hybrid)...")
        
        # Dense Vector Retriever (의미 기반) - 유사도 점수 포함
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 30,  # 상위 30개 검색 (다양성 확보)
                "score_threshold": 0.0  # 임계값 낮춤 (BM25/TF-IDF와 함께 사용되므로 낮은 점수도 허용)
            }
        )
        
        # 3-way Hybrid: Dense + BM25 + TF-IDF (직접 구현)
        self.ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever, self.tfidf_retriever],
            weights=[0.5, 0.3, 0.2]  # Dense 50%, BM25 30%, TF-IDF 20%
        )
        safe_print("✅ Ensemble Retriever 생성 완료 (Dense + BM25 + TF-IDF)")
        safe_print("   가중치: Dense 50% | BM25 30% | TF-IDF 20%")
    
    def _create_router_prompt(self):
        """Router 프롬프트 생성"""
        template = """당신은 질문을 분석하여 적절한 작업을 선택하는 라우터입니다.

질문: {question}

다음 중 하나를 선택하세요:

1. SEARCH_POLICY
   - 청년 정책 검색이 필요한 경우
   - 예: "창업 지원금", "취업 지원", "주거 지원", "대출", "교육" 등

2. GENERAL_CHAT
   - 정책과 무관한 잡담이 아니라 일반적인 인사, 감사, 대화 기록 참조 요청만 해당
   - 대화 기록 참조 요청 (이전 대화, 아까 말한 것, 처음 질문 등)
   - 예: "안녕하세요", "고맙습니다", "도움이 되었어요"
   - 예: "이전에 물어본 거 보여줘", "아까 말한 정책 뭐였지?", "맨 처음 질문 보여줘"

3. REQUEST_INFO
   - 정책 질문인데 사용자 정보(나이, 지역)가 필요한 경우

4. CLARIFY
   - 질문이 불명확하여 추가 정보가 필요한 경우
   - 예: "정책", "지원금" 같이 너무 광범위한 질문
   - **주의**: "이전", "아까", "처음" 같은 대화 참조는 GENERAL_CHAT으로 분류하세요

5. OUT_OF_SCOPE
    - 청년 정책과 무관한 일상 질문/감정 표현/잡담
    - 예 : "배고프다", "오늘 뭐 먹지?", "날씨 어때?", "심심해

**중요**:
- 청년 정책과 직접 관련이 없으면 OUT_OF_SCOPE를 선택하세요.
- 인사/감사/대화참조 요청만 GENERAL_CHAT으로 분류하세요.
- 반드시 JSON 형식으로만 답변하세요.

응답 형식:
{{
  "action": "SEARCH_POLICY",
  "reason": "창업 지원금 관련 정책 검색 필요"
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def _create_self_rag_prompt(self):
        """Self-RAG 프롬프트 생성"""
        template = """당신은 청년 정책 QA 시스템의 검증자입니다.
아래는 검색을 통해 수집된 정책 정보(context)와, 모델이 생성한 초안 답변입니다.
📋 정책 정보:
{context}
📝 모델 답변 초안:
{answer}

다음 기준으로 답변을 평가하세요:
1. 답변 내용이 위 정책 정보에 실제로 존재하는 정보에 기반하는지 확인하세요.
2. 존재하지 않는 정책명을 새로 만들어내지 않았는지 확인하세요.
3. 지원대상, 나이, 지역, 지원금액 등 주요 조건이 왜곡되지 않았는지 확인하세요.
4. {answer}가 {question}에 정확히 답변하는지 확인하세요.

반드시 아래 JSON 형식으로만 출력하세요:

{{
  "is_grounded": true or false,
  "issues": ["문제1", "문제2"],
  "suggested_fix": "문제가 있을 경우, 더 안전하고 정확한 수정 답변을 한글로 작성"
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def self_rag_verify(self, question: str, answer: str, context: str):
        """Self-RAG : 답변이 컨텍스트에 근거하는지 검증
        
        Args:
            question: 사용자 질문
            answer: 생성된 답변
            context: 검색된 정책 정보 (이미 포맷된 문자열)
        """
        try:
            chain = self.self_rag_prompt | self.llm | StrOutputParser()
            resp = chain.invoke({"context": context, "answer": answer, "question": question})
            
            # JSON만 추출
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()
            
            result = json.loads(resp)
            is_grounded = result.get("is_grounded", True)
            issues = result.get("issues", [])
            suggested_fix = result.get("suggested_fix", "")

            if is_grounded:
                safe_print("✅ Self-RAG: 근거 기반 답변으로 판단")
                return answer
            else:
                safe_print(f"⚠️ Self-RAG: 답변에 문제 발견 - {issues}")
                if suggested_fix:
                    safe_print(f"🔧 Self-RAG: 수정된 답변 사용")
                    return suggested_fix
            
            return answer
        except Exception as e:
            safe_print(f"⚠️ Self-RAG 검증 실패: {e}")
            return answer
    
    def _filter_by_category(self, question: str, docs: list):
        """질문에서 중분류 키워드를 추출하여 관련 정책만 필터링"""
        # 중분류 키워드 매핑 (실제 데이터 기반)
        category_keywords = {
            "취업": ["취업", "일자리", "채용", "구직", "면접", "입사", "재직", "고용"],
            "창업": ["창업", "스타트업", "사업", "창업자", "기업가", "자영업", "개업"],
            "전월세 및 주거급여 지원": ["주거", "주택", "전세", "월세", "임대", "보증금", "집", "거주"],
            "주택 및 거주지": ["주택", "거주", "집", "주거지"],
            "기숙사": ["기숙사", "학생숙소", "공동거주"],
            "미래역량강화": ["교육", "훈련", "학습", "강의", "수강", "자격증", "역량", "스킬"],
            "교육비지원": ["교육비", "학비", "등록금", "장학금", "수강료"],
            "취약계층 및 금융지원": ["대출", "금융", "이자", "보증", "신용", "저소득", "취약계층"],
            "건강": ["건강", "의료", "치료", "검진", "병원"],
            "문화활동": ["문화", "여행", "예술", "공연", "체험", "활동"],
            "청년참여": ["참여", "청년활동", "봉사", "위원회"],
            "정책인프라구축": ["인프라", "시스템", "플랫폼", "센터"]
        }
        
        # 질문에서 카테고리 추출
        matched_categories = []
        question_lower = question.lower()
        for category, keywords in category_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                matched_categories.append(category)
        
        # 카테고리 매칭이 없으면 필터링하지 않음 (모든 결과 반환)
        if not matched_categories:
            safe_print("  ℹ️ 특정 분야 키워드 없음, 모든 분야 검색")
            return docs
        
        safe_print(f"  🎯 매칭된 분야: {', '.join(matched_categories)}")
        
        # 중분류 필터링
        filtered = []
        for doc in docs:
            category = doc.metadata.get('중분류', '')
            # 복수 중분류 처리 (예: "취업,미래역량강화")
            if any(cat in category for cat in matched_categories):
                filtered.append(doc)
        
        # 필터링 결과가 없으면 원본 반환 (너무 엄격하지 않게)
        return filtered if filtered else docs

    def _retrieve_and_filter(self, question):
        """검색 + 메타데이터 필터링 (MultiQuery + Ensemble 사용)"""
        
        # MultiQuery: 질문을 여러 개로 확장
        if self.use_multi_query:
            queries = self.multi_query_gen.generate_queries(question)
        else:
            queries = [question]
        
        # 모든 쿼리로 검색 후 결과 통합
        all_docs = []
        seen_ids = set()
        
        for query in queries:
            try:
                # Ensemble에서 검색
                docs = self.ensemble_retriever.get_relevant_documents(query)
                
                # 중복 제거하면서 추가
                for doc in docs:
                    doc_id = doc.page_content[:100]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        
            except Exception as e:
                safe_print(f"⚠️ 쿼리 '{query}' 검색 오류: {e}")
                continue
        
        safe_print(f"🔍 총 검색 결과: {len(all_docs)}개 (중복 제거)")
        
        # 검색된 정책 목록 출력 (디버깅용)
        if all_docs:
            safe_print("📋 검색된 정책 목록:")
            for i, doc in enumerate(all_docs[:10], 1):
                policy_name = doc.metadata.get('정책명', 'N/A')
                category = doc.metadata.get('중분류', 'N/A')
                safe_print(f"  {i}. {policy_name} ({category})")
        
        # 질문에서 중분류 키워드 추출 후 필터링
        category_filtered_docs = self._filter_by_category(question, all_docs)
        if category_filtered_docs:
            safe_print(f"✅ 중분류 필터링 후: {len(category_filtered_docs)}개")
            all_docs = category_filtered_docs
        
        # 현재 날짜 기준으로 종료된 정책 필터링
        current_date = datetime.now()
        active_docs = []
        
        for doc in all_docs:
            metadata = doc.metadata
            policy_name = metadata.get('정책명', 'N/A')
            end_date_str = metadata.get('사업종료일', '')
            
            # 종료일이 없으면 포함 (상시 운영)
            if not end_date_str or end_date_str == '0':
                active_docs.append(doc)
                continue
            
            # 종료일 파싱 (YYYYMMDD 형식)
            try:
                if len(end_date_str) == 8 and end_date_str.isdigit():
                    end_date = datetime.strptime(end_date_str, '%Y%m%d')
                    
                    # 종료되지 않은 정책만 포함
                    if end_date >= current_date:
                        active_docs.append(doc)
                    else:
                        safe_print(f"  ✕ 종료된 정책: {policy_name} (종료일: {end_date_str})")
                else:
                    # 파싱 실패 시 포함
                    active_docs.append(doc)
            except:
                # 예외 발생 시 포함
                active_docs.append(doc)
        
        safe_print(f"✅ 기간 필터링 후: {len(active_docs)}개 (종료된 정책 제외)")
        
        # 사용자 정보가 없으면 기간 필터링만 적용하고 더 많은 결과 반환
        if not (self.user_age or self.user_region or self.user_education):
            return active_docs[:10]  # 5개 → 10개로 증가 (다양성 확보)
        
        # 나이/지역/학력 필터링 시작
        filtered_docs = []
        for doc in active_docs:
            metadata = doc.metadata
            
            # 나이 필터링
            age_match = True
            if self.user_age:
                try:
                    min_age = int(metadata.get('지원최소연령', '0') or '0')
                    max_age = int(metadata.get('지원최대연령', '0') or '0')
                    
                    if min_age > 0 and self.user_age < min_age:
                        age_match = False
                    if max_age > 0 and max_age < 999 and self.user_age > max_age:
                        age_match = False
                except:
                    pass
            
            # 학력 필터링 (학력요건 필드 확인)
            education_match = True
            if self.user_education:
                edu_requirement = metadata.get('학력요건', '')
                
                # "제한없음", "기타" 또는 빈 값이면 모두 통과
                if not edu_requirement or '제한없음' in edu_requirement or '기타' in edu_requirement:
                    education_match = True
                else:
                    # 사용자 학력에 따른 매칭 키워드
                    user_edu_match = False
                    
                    if "중학교" in self.user_education:
                        # 중학교는 중학 이상만 매칭
                        user_edu_match = any(kw in edu_requirement for kw in ["중학", "중졸"])
                    
                    elif "고등학교" in self.user_education:
                        # 고등학교는 고등학교 이상 (대학 미만)
                        user_edu_match = any(kw in edu_requirement for kw in ["고등", "고졸", "고교"])
                        # 대학 요건이 있으면 불통과
                        if any(kw in edu_requirement for kw in ["대학", "학사", "석", "박사"]):
                            user_edu_match = False
                    
                    elif "대학교" in self.user_education:
                        # 대학교 재학/졸업은 대학 관련 모두 매칭
                        user_edu_match = any(kw in edu_requirement for kw in ["대학", "학사", "재학", "졸업"])
                        # 대학원만 요구하면 불통과
                        if ("석" in edu_requirement or "박사" in edu_requirement) and "대학" not in edu_requirement:
                            user_edu_match = False
                    
                    elif "대학원" in self.user_education:
                        # 대학원은 모든 학력 요건 통과 (최고 학력)
                        user_edu_match = True
                    
                    education_match = user_edu_match
            
            # 지역 필터링 (계층적 매칭: 전국 → 시/도 → 시/군/구)
            region_match = True
            if self.user_region:
                org_name = metadata.get('주관기관명', '')
                registered_org = metadata.get('등록기관명', '')  # 실제 지역 정보
                upper_org = metadata.get('상위등록기관명', '')  # 시/도 정보
                additional_cond = metadata.get('추가자격조건', '')
                reg_group = metadata.get('재공기관그룹', '')
                
                policy_name = metadata.get('정책명', 'N/A')
                
                # 1순위: 전국 정책은 항상 포함
                if '중앙부처' in reg_group or '전국' in org_name:
                    region_match = True
                    safe_print(f"  ✓ 전국 정책: {policy_name} (기관: {org_name})")
                else:
                    # 2순위: 시/도 단위 매칭 (구/군 입력 시에도 시/도 정책 포함)
                    sido_list = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                               '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
                    
                    user_sido = None
                    for sido in sido_list:
                        if sido in self.user_region:
                            user_sido = sido
                            break
                    
                    # 시/도 매칭 확인 (등록기관명과 상위등록기관명도 확인)
                    if user_sido and (user_sido in org_name or user_sido in registered_org or user_sido in upper_org):
                        region_match = True
                        safe_print(f"  ✓ 시/도 매칭: {policy_name} (시/도: {user_sido}, 등록: {registered_org})")
                    else:
                        # 3순위: 구/군 단위 상세 매칭
                        # user_region 형식: "서울특별시 구로구" 또는 "경기도 의정부시"
                        user_region_tokens = []
                        if user_sido:
                            user_region_tokens.append(user_sido)
                        
                        # 공백으로 분리해서 마지막 토큰이 구/군/시
                        region_parts = self.user_region.split()
                        if len(region_parts) > 1:
                            district = region_parts[-1]  # 마지막 부분이 구/군/시
                            user_region_tokens.append(district)
                        
                        region_match = False
                        for token in user_region_tokens:
                            # 주관기관명, 등록기관명, 추가자격조건 모두 확인
                            if token in org_name or token in registered_org or token in additional_cond:
                                region_match = True
                                safe_print(f"  ✓ 상세 매칭: {policy_name} (토큰: {token}, 등록: {registered_org})")
                                break
                        
                        if not region_match:
                            safe_print(f"  ✗ 제외: {policy_name} (등록: {registered_org}, 주관: {org_name})")
            
            # 세 조건 모두 만족하면 포함
            if age_match and region_match and education_match:
                filtered_docs.append(doc)
        
        safe_print(f"✅ 필터링 후: {len(filtered_docs)}개")
        
        # 결과가 너무 적으면 전국 정책만이라도 반환
        if len(filtered_docs) < 5:
            safe_print("⚠️ 필터링 결과 부족, 전국 정책 추가 검색")
            for doc in active_docs:
                if len(filtered_docs) >= 10:
                    break
                metadata = doc.metadata
                reg_group = metadata.get('재공기관그룹', '')
                if '중앙부처' in reg_group and doc not in filtered_docs:
                    filtered_docs.append(doc)
        
        return filtered_docs[:10]  # 상위 10개 반환 (더 다양한 선택지)
    
    def _format_docs(self, docs):
        """문서 포맷팅"""
        if not docs:
            return "검색된 정책이 없습니다."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            
            # 지원금액 표시
            min_amount = metadata.get('최소지원금액', '0')
            max_amount = metadata.get('최대지원금액', '0')
            if min_amount == '0' and max_amount == '0':
                amount_str = "정보 없음"
            elif min_amount == max_amount:
                amount_str = f"{int(min_amount):,}원"
            else:
                amount_str = f"{int(min_amount):,}원 ~ {int(max_amount):,}원"
            
            # 신청기간 표시
            apply_period = metadata.get('신청기간', '')
            if apply_period:
                apply_str = apply_period.replace('~', ' ~ ')
            else:
                apply_str = "상시 신청"
            
            # 학력요건 표시
            edu_req = metadata.get('학력요건', '제한없음')
            
            formatted.append(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 정책 {i}. {metadata.get('정책명', 'N/A')}

🏢 담당기관: {metadata.get('주관기관명', 'N/A')}
📂 분야: {metadata.get('대분류', 'N/A')} > {metadata.get('중분류', 'N/A')}

👥 지원대상
  • 연령: {metadata.get('지원최소연령', 'N/A')}세 ~ {metadata.get('지원최대연령', 'N/A')}세
  • 학력: {edu_req}
  • 거주지: {metadata.get('등록기관명', '전국')}

💰 지원내용
  • 지원금액: {amount_str}
  • 신청기간: {apply_str}

📝 상세설명
{metadata.get('지원내용', doc.page_content[:300])}

🔗 참고링크: {metadata.get('참고URL1', '정보 없음')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
        return "\n".join(formatted)
    
    def _format_chat_history(self) -> str:
        """self.chat_history(HumanMessage/AIMessage 리스트)를 사람이 읽기 좋은 문자열로 변환"""
        if not self.chat_history:
            return ""
        
        lines = []
        for msg in self.chat_history:
            role = "사용자" if isinstance(msg, HumanMessage) else "상담사"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def _run_multi_agent_mode(self, question: str):
    # 라우터 재사용
        routing_result = self.route_query(question)
        action = routing_result.get('action')

        if action == "SEARCH_POLICY":
            # 기존 GENERAL_CHAT 분기 로직을 그대로 쓰거나
            # query() 안의 프롬프트를 그대로 복사해서 사용
            chat_history_txt = self._format_chat_history()
            prompt = ChatPromptTemplate.from_template(
                """당신은 친근한 청년 정책 상담사입니다.
                [대화 기록]
                {chat_history}
                [사용자 질문]
                {question}
                답변:"""
            )
            return (prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": chat_history_txt, "question": question}
            )
        
        elif action == "REQUEST_INFO":
            return """더 정확한 정책을 추천해드리기 위해 정보가 필요합니다! 😊
            1. 나이
            2. 지역 (예: 서울특별시, 경기도 의정부시)
            정보를 입력해주시면 맞춤형 정책을 찾아드리겠습니다!"""
        
        elif action == "CLARIFY":
            return """질문을 조금만 더 구체적으로 말씀해주시겠어요? 😊
            예) 취업/창업/주거/금융/교육 중 어떤 분야가 궁금하신가요?"""
        
        # SEARCH_POLICY 기본 흐름은 기존 내부 메서드 재사용
        docs = self._retrieve_and_filter(question)
        context = self._format_docs(docs)
        chat_history_txt = self._format_chat_history()

        prompt = ChatPromptTemplate.from_template(
            """당신은 청년 정책 전문 상담사입니다.
            [대화 기록]
            {chat_history}
            [정책 정보]
            {context}
            [사용자 질문]
            {question}
            답변:"""
        )

        raw_answer = (prompt | self.llm | StrOutputParser()).invoke(
            {"chat_history": chat_history_txt,
             "context": context,
             "question": question}
        )
        return self.self_rag_verify(question, raw_answer, context)
    

    def query(self, question: str):
        """
        질문에 답변 (Router + 대화 메모리 + Self-RAG 적용)
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 답변
        """
        user_info = ""
        if self.user_age or self.user_region:
            user_info = f" (나이: {self.user_age}세, 지역: {self.user_region})"
        
        safe_print(f"\n🔍 질문: {question}{user_info}")
        
        # (추가) 멀티 에이전트 모드 사용 시
        if getattr(self, "use_multi_agent", False):
            answer = self.multi_agent.run(question)

            # 기존 메모리 저장 로직은 그대로 재사용
            if self.chat_history is not None and answer:
                self.chat_history.append(HumanMessage(content=question))
                self.chat_history.append(AIMessage(content=answer))

            return answer

        # 1단계: Router로 질문 분석
        routing_result = self.route_query(question)
        action = routing_result.get('action')
        answer = ""
        
        # 2단계: Action에 따라 처리
        if action == "GENERAL_CHAT":
            safe_print("💬 일반 대화 모드\n")
            prompt = ChatPromptTemplate.from_template(
                """당신은 친근한 청년 정책 상담사입니다.
                아래는 지금까지의 대화 기록입니다.
                
                [대화 기록]
                {chat_history}
                
                [사용자 질문]
                {question}

                답변 가이드:
                1. 사용자가 "이전에 물어본 것", "아까 말한 정책" 등을 언급하면 대화 기록을 참조하세요.
                2. 대화 기록에 정책명이나 구체적 정보가 있다면 그대로 인용하세요.
                3. 정책 상세 정보가 필요하면 "다시 검색해드릴까요?"라고 물어보세요.
                4. 일반적인 인사나 감사는 간단하고 따뜻하게 답변하세요.

                답변:"""
            )
            chat_history_txt = self._format_chat_history()
            answer = (prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": chat_history_txt, "question": question})
        
        elif action == "REQUEST_INFO":
            safe_print("📋 사용자 정보 필요\n")
            answer = """더 정확한 정책을 추천해드리기 위해 정보가 필요합니다! 😊

다음 정보를 알려주시겠어요?
1. 나이: 만 몇 세이신가요?
2. 지역: 어디에 거주하시나요? (예: 서울특별시, 경기도 의정부시)

정보를 입력하시면 맞춤형 정책을 찾아드리겠습니다!"""
        
        elif action == "CLARIFY":
            safe_print("❓ 질문 명확화 필요\n")
            answer = """질문을 좀 더 구체적으로 말씀해주시겠어요? 😊

예를 들면:
- "창업 지원금이 궁금해요"
- "청년 취업 지원 프로그램 알려주세요"
- "전월세 대출 정책이 있나요?"

구체적인 분야를 말씀해주시면 더 정확한 정책을 찾아드릴게요!"""
        elif action == "OUT_OF_SCOPE":
            safe_print("🚫 정책 범위 외 질문\n")
            answer = """저는 청년 정책 상담에 집중하는 챗봇입니다😊
            청년 정책과 관련된 질문을 해주시면 더 정확히 도와드릴게요!
            
            예시:
            - "청년 취업 지원 프로그램 알려주세요"
            - "전월세 보증금/월세 지원 정책이 있나요?"
            - "청년 창업 지원금 조건이 궁금해요"
            - "자격증/교육비 지원 정책 추천해주세요"

            원하시면 '취업/창업/주거/교육/금융' 중 관심 분야를 말씀해 주셔도 됩니다!
            """
            
        else:  # SEARCH_POLICY
            safe_print("⏳ 정책 검색 중...\n")
            # 1) 문서 검색
            docs = self._retrieve_and_filter(question)
            # 2) 컨텍스트 포매팅
            context = self._format_docs(docs)
            # 3) 대화 기록
            chat_history_txt = self._format_chat_history()

            # 4) 1차 답변 생성 (대화 기록 + 컨텍스트 같이 제공)
            prompt = ChatPromptTemplate.from_template("""당신은 청년 정책 전문 상담사입니다.
            아래는 지금까지의 대화 기록과, 검색된 정책 정보입니다.
            
            [대화 기록]
            {chat_history}
            
            [정책 정보]
            {context}

            [사용자 질문]
            {question}
            
            답변 가이드라인:
            1. **검색된 모든 정책을 빠짐없이 소개**하세요.
            2. 각 정책마다 다음 정보를 **원본 그대로** 포함하세요:
               - 정책명
               - 담당기관
               - 지원대상 (연령, 학력, 거주지)
               - 지원내용 (구체적인 금액, 지원 방식)
               - 신청기간
               - 참고링크
            3. 정보를 요약하거나 생략하지 마세요. **제공된 정보를 그대로 전달**하세요.
            4. 정책 정보에 없는 내용은 추가하지 마세요.
            5. 친근하고 격려하는 톤으로 작성하되, **정보는 정확하고 상세하게** 제공하세요.
            6. 각 정책 사이에 구분선(━━━)을 넣어 읽기 쉽게 하세요.
            7. 연령이 0세 ~ 0세인 경우 "제한없음"으로 표현하세요.
            8. 연령이 n세 ~ 0세인 경우 "n세 이상"으로 표현하세요.
                                                      
            답변:"""
            )
            raw_answer = (prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": chat_history_txt,
                 "context": context,
                 "question": question})
            
            # 5) Self-RAG 검증
            answer = self.self_rag_verify(question, raw_answer, context)
        
        # 3단계: 대화 메모리에 저장
        if self.chat_history is not None and answer:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
        
        return answer
    
    def set_user_info(self, age=None, region=None, education=None):
        """
        사용자 정보 설정
        
        Args:
            age: 나이
            region: 지역 (예: "경기도 의정부시")
            education: 학력 (예: "대학교 재학", "고등학교 졸업")
        """
        self.user_age = age
        self.user_region = region
        self.user_education = education
        
        info = []
        if age:
            info.append(f"나이 {age}세")
        if region:
            info.append(f"지역 {region}")
        if education:
            info.append(f"학력 {education}")
        
        if info:
            safe_print(f"✅ 사용자 정보 설정: {', '.join(info)}", force=True)
            safe_print(f"   → 전구/중앙부처 정책 + {region} 정책이 함께 검색됩니다.", force=True)
    
    def route_query(self, question: str):
        """
        질문을 분석하여 적절한 작업으로 라우팅
        
        Args:
            question: 사용자 질문
            
        Returns:
            dict: 라우팅 결과
        """
        try:
            # Router LLM 호출
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            response = router_chain.invoke({"question": question})
            
            # JSON 파싱
            # 응답에서 JSON 부분만 추출 (```json...``` 제거)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            
            # REQUEST_INFO인 경우, 사용자 정보가 이미 있으면 SEARCH_POLICY로 변경
            if result.get('action') == 'REQUEST_INFO':
                if self.user_age or self.user_region:
                    safe_print(f"ℹ️  사용자 정보 이미 있음 (나이: {self.user_age}, 지역: {self.user_region})")
                    result['action'] = 'SEARCH_POLICY'
                    result['reason'] = '사용자 정보 있음, 정책 검색 진행'
            
            safe_print(f"🎯 라우팅 결과: {result['action']} - {result.get('reason', '')}")
            
            return result
            
        except Exception as e:
            safe_print(f"⚠️ 라우팅 오류: {e}, 기본 검색으로 진행")
            return {
                "action": "SEARCH_POLICY",
                "reason": "라우팅 실패, 기본 검색"
            }
        
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "=" * 70)
        print("💬 청년 정책 상담 챗봇")
        print("=" * 70)
        
        # 사용자 정보 입력
        print("\n👤 사용자 정보를 입력해주세요 (Enter로 건너뛰기 가능)")
        
        age_input = input("나이: ").strip()
        if age_input:
            try:
                self.user_age = int(age_input)
            except:
                print("⚠️ 유효하지 않은 나이입니다.")
        
        region_input = input("지역 (예: 경기도 의정부시, 서울특별시): ").strip()
        if region_input:
            self.user_region = region_input
        
        if self.user_age or self.user_region:
            self.set_user_info(self.user_age, self.user_region)
        
        print("\n질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
        
        while True:
            try:
                question = input("👤 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 상담을 종료합니다. 감사합니다!")
                    break
                
                if not question:
                    continue
                
                # 답변 생성 (Self-RAG + 대화 메모리 적용)
                answer = self.query(question)
                print(f"\n🤖 답변:\n{answer}\n")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")


def main():
    """메인 함수"""
    # RAG 시스템 초기화
    rag = YouthPolicyRAG()
    
    # 대화형 모드 실행
    rag.interactive_mode()


if __name__ == "__main__":
    main()
