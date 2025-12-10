"""
고급 RAG 파이프라인 - 청년 정책 챗봇
- Router: 질문 검증 및 정제
- RegionFilter: 지역 기반 필터링
- MultiQueryGenerator: 다중 관점 쿼리 생성
- EnsembleRetriever: Dense + BM25 검색
- ReciprocalRankFusion: 검색 결과 통합
- ConversationMemory: 대화 맥락 관리
- AdvancedRAGPipeline: 통합 파이프라인
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import warnings

# TensorFlow 로그 억제
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

# BM25, Ensemble Retriever
try:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
    RETRIEVERS_AVAILABLE = True
except ImportError:
    RETRIEVERS_AVAILABLE = False
    BM25Retriever = None
    EnsembleRetriever = None

# 환경 변수 로드
load_dotenv()


# ============================================================================
# 2. Router: 질문 검증 및 정제
# ============================================================================

class QueryRouter:
    """사용자 쿼리를 검증하고 정제하는 라우터"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.router_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자 질문을 분석하고 정제하는 라우터입니다.

            작업:
            1. 질문이 의미 있는지 검증 (인사말, 욕설, 무의미한 입력 제외)
            2. 질문 카테고리 분류 (정책검색, 추천, 일반질문 등)
            3. LLM이 처리하기 좋은 형태로 정제
            4. 지역이 없으면 사용자에게 다시 입력하도록 유도
            
            응답 형식 (JSON):
            {{
                "is_valid": true/false,
                "category": "정책검색|정책추천|일반질문|기타",
                "refined_query": "정제된 질문",
                "reason": "판단 이유"
            }}"""),
            ("user", "{query}")
        ])
    
    def route(self, query: str) -> Dict:
        """쿼리를 검증하고 정제"""
        try:
            response = self.router_prompt | self.llm | StrOutputParser()
            result_str = response.invoke({"query": query})
            
            # JSON 파싱
            result = json.loads(result_str)
            
            return result
        except Exception as e:
            return {
                "is_valid": True,
                "category": "일반질문",
                "refined_query": query,
                "reason": "파싱 실패로 원본 사용"
            }


# ============================================================================
# 3. RegionFilter: 지역 필터링 유틸리티
# ============================================================================

class RegionFilter:
    """지역 기반 필터링을 수행하는 유틸리티 클래스"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.region_detection_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자 질문에서 지역 정보를 추출하는 전문가입니다.

작업:
1. 질문에서 지역명(시/도, 시/군/구)을 추출
2. '전체', '전국', '모든', '모두' 등의 키워드가 있으면 '전국'으로 분류
3. 지역명이 없으면 '전국'으로 분류

응답 형식 (JSON):
{{
    "has_region": true/false,
    "is_national": true/false,
    "region_name": "지역명 또는 null",
    "reason": "판단 이유"
}}

예시:
- "대구 월세 지원" -> {{"has_region": true, "is_national": false, "region_name": "대구", "reason": "대구 지역 명시"}}
- "전국 청년 정책" -> {{"has_region": true, "is_national": true, "region_name": null, "reason": "전국 키워드 사용"}}
- "월세 지원" -> {{"has_region": True, "is_national": True, "region_name": null, "reason": "지역 미 명시로 전국 기본 적용"}}
"""),
            ("user", "{query}")
        ])
    
    def detect_region(self, query: str) -> Dict:
        """질문에서 지역 정보 추출"""
        try:
            response = self.region_detection_prompt | self.llm | StrOutputParser()
            result_str = response.invoke({"query": query})
            
            # JSON 파싱
            result = json.loads(result_str)
            
            return result
        except Exception as e:
            return {
                "has_region": True,
                "is_national": True,
                "region_name": None,
                "reason": "파싱 실패로 전국 기본 적용"
            }
    
    def build_filter(self, region_info: Dict) -> Optional[Dict]:
        """지역 정보를 바탕으로 메타데이터 필터 생성"""
        if region_info.get('is_national', False):
            # 전국 정책만 필터링
            return {"지역범위": "전국"}
        elif region_info.get('has_region', False) and region_info.get('region_name'):
            # 특정 지역이 포함된 정책 필터링 (ChromaDB는 $or 미지원, Python에서 후처리)
            # ChromaDB 필터로는 우선 모든 정책을 가져오고, 후처리에서 필터링
            return None  # 필터 없이 검색 후 Python에서 필터링
        else:
            # 필터 없음 (모든 정책 검색)
            return None
    
    def filter_documents(self, documents: List, region_info: Dict) -> List:
        """검색된 문서를 지역 정보로 후처리 필터링"""
        # 전국 검색인 경우: 지역범위가 "전국"인 문서만 필터링
        if region_info.get('is_national', False):
            filtered_docs = []
            for doc in documents:
                if doc.metadata.get('지역범위') == '전국':
                    filtered_docs.append(doc)
            return filtered_docs if filtered_docs else documents
        
        # 특정 지역 검색인 경우: 해당 지역 + 전국 정책 포함
        if not region_info.get('has_region', False):
            return documents
            
        region_name = region_info.get('region_name')
        if not region_name:
            return documents
        
        filtered_docs = []
        for doc in documents:
            # 전국 정책은 항상 포함
            if doc.metadata.get('지역범위') == '전국':
                filtered_docs.append(doc)
            # 지역 필드에 해당 지역명이 포함되어 있으면 포함
            elif region_name in doc.metadata.get('지역', ''):
                filtered_docs.append(doc)
        
        return filtered_docs if filtered_docs else documents
    

    


# ============================================================================
# 4. Multi-Query Generator: 다중 관점 쿼리 생성
# ============================================================================

class MultiQueryGenerator:
    """하나의 질문을 여러 관점의 쿼리로 확장"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.multi_query_prompt = ChatPromptTemplate.from_messages([
            "system", """당신은 사용자의 원본 질문을 **의도와 핵심 키워드를 유지**한 채 검색에 최적화된 여러 관점의 쿼리로 확장하는 전문가입니다.

            **원본 질문의 내용이나 조건을 임의로 추가하거나 변경하지 마세요. 오직 검색 관점만 다양화해야 합니다.

            주어진 질문을 3가지 다른 관점의 검색 쿼리로 재구성하세요:

            1.  **지역(Region) 추출 강제: 사용자가 지역을 언급하면, 해당 지역에 집중해
            2.  **정책 키워드(Policy Keyword): 질문의 **핵심 의도**와 관련된 정책 키워드를 추출하여 관련된 정책만 반환할 것.(예: "취업 면접 수당" -> "청년 구직 활동 지원금", "면접비 지원 사업", "취업 역량 강화 프로그램" 등 유의어 포함 ) 관련 정책만 반환)
            3.  **유사한 의미 또는 관련 정책명**을 포함하는 쿼리 (유의어 활용)

            각 쿼리는 한 줄로 작성하고, 번호 없이 줄바꿈(\n)으로 구분하세요.""",
            ("user", "{query}")
        ])
    
    def generate(self, query: str) -> List[str]:
        """다중 쿼리 생성"""
        try:
            response = self.multi_query_prompt | self.llm | StrOutputParser()
            result = response.invoke({"query": query})
            
            # 쿼리 분리 (줄바꿈 기준)
            queries = [q.strip() for q in result.split('\n') if q.strip()]
            # 원본 쿼리 포함
            all_queries = [query] + queries
            
            
            return all_queries
        
        except Exception as e:
            return [query]


# ============================================================================
# 5. Ensemble Retriever: Dense + BM25
# ============================================================================

class EnsembleRetriever:
    """Dense, BM25 검색을 결합한 앙상블 리트리버"""
    
    def __init__(
        self, 
        documents: List[any],
        vectorstore: Chroma,
        llm: ChatOpenAI = None,
        bm25_k: int = 5,
        vector_k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        self.documents = documents
        self.vectorstore = vectorstore
        self.llm = llm
        
        # 파라미터 저장
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # 각 리트리버 초기화
        self._build_bm25()
        self._build_vector()
    
    def _build_bm25(self):
        """BM25 Retriever 생성"""
        if not RETRIEVERS_AVAILABLE or BM25Retriever is None:
            self.bm25_retriever = None
            return
        
        if not self.documents:
            self.bm25_retriever = None
            return
        
        try:
            # BM25Retriever 초기화 (from_documents 사용)
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=self.documents,
                k=self.bm25_k
            )
        except TypeError as e:
            # from_documents가 실패하면 직접 초기화 시도
            try:
                self.bm25_retriever = BM25Retriever(docs=self.documents)
                self.bm25_retriever.k = self.bm25_k
            except Exception as e2:
                self.bm25_retriever = None
        except Exception as e:
            self.bm25_retriever = None
    
    def _build_vector(self):
        """Vector Retriever 생성"""
        try:
            # VectorStore 상태 확인
            test_search = self.vectorstore.similarity_search("테스트", k=1)
            
            self.vector_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.vector_k}
            )
        except Exception as e:
            self.vector_retriever = None
    
    def dense_search(self, query: str, metadata_filter: Dict = None) -> List[Tuple[any, float]]:
        """Dense 검색 (임베딩 기반)"""
        try:
            if self.vector_retriever:
                if metadata_filter:
                    # 메타데이터 필터 적용
                    docs = self.vectorstore.similarity_search(
                        query, 
                        k=self.vector_k,
                        filter=metadata_filter
                    )
                else:
                    docs = self.vector_retriever.invoke(query)
                
                results = [(doc, 1.0) for doc in docs]
                return results
            return []
        except Exception as e:
            return []
    
    def bm25_search(self, query: str) -> List[Tuple[any, float]]:
        """BM25 검색 (키워드 기반)"""
        try:
            if self.bm25_retriever:
                docs = self.bm25_retriever.invoke(query)
                results = [(doc, 1.0) for doc in docs]
                return results
            return []
        except Exception as e:
            return []
    
    def retrieve(self, queries: List[str], metadata_filter: Dict = None) -> Dict[str, List[Tuple[any, float]]]:
        """모든 검색 전략 실행"""
        all_results = {
            'dense': [],
            'bm25': []
        }
        
        for query in queries:
            all_results['dense'].extend(self.dense_search(query, metadata_filter))
            all_results['bm25'].extend(self.bm25_search(query))
        
        return all_results
    
    def get_ensemble(self, query: str) -> List[any]:
        """Ensemble 검색 (가중치 적용)"""
        if not RETRIEVERS_AVAILABLE or EnsembleRetriever is None:
            return self.dense_search(query)
        
        try:
            retrievers = []
            weights = []
            
            if self.bm25_retriever:
                retrievers.append(self.bm25_retriever)
                weights.append(self.bm25_weight)
            
            if self.vector_retriever:
                retrievers.append(self.vector_retriever)
                weights.append(self.vector_weight)
            
            if not retrievers:
                return []
            
            # 가중치 정규화
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # LangChain의 EnsembleRetriever 사용
            ensemble = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights
            )
            
            docs = ensemble.invoke(query)
            return docs
            
        except Exception as e:
            return []


# ============================================================================
# 6. RRF (Reciprocal Rank Fusion): 검색 결과 통합
# ============================================================================

class ReciprocalRankFusion:
    """여러 검색 결과를 랭킹 기반으로 통합"""
    
    def __init__(self, k: int = 60):
        self.k = k  # RRF 상수
    
    def fuse(self, results_dict: Dict[str, List[Tuple[any, float]]], top_k: int = 10) -> List[any]:
        """RRF로 결과 통합"""
        doc_scores = {}
        
        for method, results in results_dict.items():
            for rank, (doc, score) in enumerate(results, 1):
                doc_id = doc.metadata.get('policy_id', id(doc))
                
                # RRF 점수 계산: 1 / (k + rank)
                rrf_score = 1.0 / (self.k + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'doc': doc, 'score': 0}
                doc_scores[doc_id]['score'] += rrf_score
        
        # 점수 기준 정렬
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        final_docs = [item[1]['doc'] for item in sorted_docs[:top_k]]
        
        return final_docs


# ============================================================================
# 7. Memory Store: 대화 맥락 관리
# ============================================================================

@dataclass
class ConversationMemory:
    """대화 기록 관리"""
    messages: List[Dict] = field(default_factory=list)
    max_history: int = 10
    
    def add_message(self, role: str, content: str):
        """메시지 추가"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 최대 기록 수 제한
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_context(self) -> str:
        """대화 맥락 문자열 생성"""
        if not self.messages:
            return "이전 대화 없음"
        
        context_parts = []
        for msg in self.messages[-6:]:  # 최근 3턴
            role = "사용자" if msg['role'] == 'user' else "AI"
            context_parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """기록 초기화"""
        self.messages.clear()


# ============================================================================
# 8. Advanced RAG Pipeline: 통합 파이프라인
# ============================================================================

class AdvancedRAGPipeline:
    """고급 RAG 파이프라인"""
    
    def __init__(
        self,
        documents: List[any],
        vectorstore: Chroma,
        llm: ChatOpenAI,
        enable_router: bool = True,
        enable_multi_query: bool = True,
        enable_ensemble: bool = True,
        enable_rrf: bool = True,
        enable_memory: bool = True,
        enable_region_filter: bool = True,
        bm25_k: int = 5,
        vector_k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        self.documents = documents
        self.vectorstore = vectorstore
        self.llm = llm
        
        # 각 컴포넌트 초기화
        self.router = QueryRouter(llm) if enable_router else None
        self.multi_query = MultiQueryGenerator(llm) if enable_multi_query else None
        self.region_filter = RegionFilter(llm) if enable_region_filter else None
        self.ensemble = EnsembleRetriever(
            documents=documents,
            vectorstore=vectorstore,
            llm=llm,
            bm25_k=bm25_k,
            vector_k=vector_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        ) if enable_ensemble else None
        self.rrf = ReciprocalRankFusion() if enable_rrf else None
        self.memory = ConversationMemory() if enable_memory else None
        
        # 최종 답변 생성 프롬프트
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 '온통청년 청년정책 전담 챗봇 선배봇'입니다.

역할:
- 너는 청년 정책(특히 주거·월세·일자리·복지) 정보를, 사용자가 이해하기 쉽게 정리해 주는 선배야.
- 후배에게 알려주듯 친근하고 부드러운 말투로 답변해. (예: ~해줄게, ~해보자, ~이야)
- 반드시 '검색된 정책 정보(documents)' 안에 있는 내용만 사용해서 답변해.
- 추측하거나 지어내지 말고, 문서에 없는 정보는 "문서에 없는 내용이라 확답이 어렵다"고 말해.

출력 형식(꼭 지켜야 함):

1. 항상 아래 문구로 시작한다.
    안녕! 나는 청년들의 든든한 정책 선배, 청년이음 선배봇🌟이야.
    주거, 월세, 일자리, 복지 정책 등 궁금한 점이 있으면 언제든지 나에게 물어봐!😺

2. 그 다음 줄에 질문을 그대로 보여준다.
   사용자 질문 : {query}

3. 그 다음에 '답변 :'을 쓰고, 정책을 번호를 매겨서 정리한다.
   - 최소 3개, 최대 5개의 정책을 선택해서 답변해.
   - 각 정책은 아래 구조를 따른다.

   예시 형식:

   답변 :
   1. 정책명 (주체/지역)
             
    🔸사업 개요
        - 사업 기간 : ...
        - 목적 : ...

    🔸신청 자격(핵심 요건)
        - 연령 : ...
        - 주거 : ...
        - 소득 : ...
        - 기타 조건 : ...

    🔸지원 금액·기간
        - 월 지원 금액 : ...
        - 지원 기간 : ...

    🔸신청 방법(절차)
        - 어디에 신청 : ...
        - 어떻게 신청 : ...

   2. 정책명 (주체/지역)
             
    🔸사업 개요
        - 사업 기간 : ...
        - 목적 : ...

    🔸신청 자격(핵심 요건)
        - 연령 : ...
        - 주거 : ...
        - 소득 : ...
        - 기타 조건 : ...

    🔸지원 금액·기간
        - 월 지원 금액 : ...
        - 지원 기간 : ...

    🔸신청 방법(절차)
        - 어디에 신청 : ...
        - 어떻게 신청 : ...
   3. ...
             
    4. 마지막에 '출처' 블록을 적는다.
    - 문서 메타데이터(파일명, 페이지 정보 등)가 있으면 최대한 활용해서 작성한다.
    - 예시:
    🔹 출처:
        - 온통청년 청년정책 포털 (https://youthcenter.go.kr)

작성 시 유의사항:
- 정책명이 같은 것을 중복해서 쓰지 말 것.
- 질문에서 언급한 지역(예: 경북, 대구 등)과 직접적으로 관련 있는 정책을 최우선으로 선택할 것.
- 질문에서 '월세', '보증금', '전세' 등 키워드가 나오면, 주거·월세 관련 정책 위주로 정리할 것.
- 숫자(지원 금액, 기간, 연령)는 가능한 한 구체적인 값으로 써 줄 것.
- 줄바꿈이나 문단 구분을 명확히 해서 가독성을 높일 것.

"""),
            ("user", """[대화 맥락]
{context}

[검색된 정책 정보]
{documents}

[현재 질문]
{query}""")
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 청년 정책 상담 답변을 짧게 요약하는 보조 도우미입니다.

목표:
- 사용자의 이해를 돕기 위해, 위에서 생성된 긴 답변을 핵심만 한 번 더 정리해.
- 정책명, 대상(누가 받을 수 있는지), 지원 유형/금액 정도만 담아 한두 단락으로 요약해.
- 새로운 정보를 만들지 말고, 반드시 이미 주어진 답변 내용만 재구성해.
- 기호들 특히(**)는 삭제하고 깔끔한 문장으로 작성해.
- 번호를 매겨서 가독성이 좋게 작성해.
"""),
            ("user", """다음 답변을 사용자가 빠르게 이해할 수 있도록 핵심만 요약해줘.

[전체 답변]
{answer}
""")
        ])

    def query(self, user_query: str) -> Dict:
        """전체 파이프라인 실행"""
        
        # 1. Router: 질문 검증 및 정제
        if self.router:
            route_result = self.router.route(user_query)
            if not route_result['is_valid']:
                return {
                    "answer": "음… 이해를 못 했어😥. 한 번만 다시 얘기해줘!",
                    "documents": [],
                    "metadata": route_result
                }
            query = route_result['refined_query']
        else:
            query = user_query
        
        # 2. Region Filter: 지역 정보 추출 및 필터 생성
        metadata_filter = None
        region_info = None
        if self.region_filter:
            region_info = self.region_filter.detect_region(query)
            metadata_filter = self.region_filter.build_filter(region_info)
        
        # 3. Multi-Query: 다중 쿼리 생성
        if self.multi_query:
            queries = self.multi_query.generate(query)
        else:
            queries = [query]
        
        # 4. Ensemble Retriever: 다중 검색 (메타데이터 필터 적용)
        if self.ensemble:
            search_results = self.ensemble.retrieve(queries, metadata_filter)
        else:
            if metadata_filter:
                docs_with_score = self.vectorstore.similarity_search_with_score(
                    query, k=5, filter=metadata_filter
                )
            else:
                docs_with_score = self.vectorstore.similarity_search_with_score(query, k=5)
            search_results = {'dense': docs_with_score}
        
        # 5. RRF: 검색 결과 통합
        if self.rrf:
            docs = self.rrf.fuse(search_results, top_k=20)
        else:
            docs = [doc for doc, score in search_results['dense']]
        
        # 6. Region Filter: 지역 기반 후처리 필터링
        if self.region_filter and region_info:
            docs = self.region_filter.filter_documents(docs, region_info)
        
        # 7. Memory: 대화 맥락 가져오기
        if self.memory:
            context = self.memory.get_context()
        else:
            context = "이전 대화 없음"
        
        # 8. LLM: 최종 답변 생성 (정책명 중복 없이 최대 3개만)
        seen_titles = set()
        unique_docs = []
        for doc in docs:
            title = doc.metadata.get('정책명', '제목 없음')
            if title not in seen_titles:
                seen_titles.add(title)
                unique_docs.append(doc)
            if len(unique_docs) >= 3:
                break
        docs_text = "\n\n".join([
            f"[정책 {i+1}] {doc.metadata.get('정책명', '제목 없음')}\n{doc.page_content[:500]}"
            for i, doc in enumerate(unique_docs)
        ])
        
        try:
            response = self.answer_prompt | self.llm | StrOutputParser()
            answer = response.invoke({
                "context": context,
                "documents": docs_text,
                "query": user_query
            })
            
            # 9. 요약 생성
            summary_response = self.summary_prompt | self.llm | StrOutputParser()
            summary = summary_response.invoke({"answer": answer})
            
            # 메모리에 저장
            if self.memory:
                self.memory.add_message("user", user_query)
                self.memory.add_message("assistant", answer)
            
            return {
                "answer": answer,
                "summary": summary,
                "documents": docs,
                "metadata": {
                    "queries": queries,
                    "num_docs_retrieved": len(docs),
                    "has_context": bool(self.memory and self.memory.messages),
                    "region_filter": metadata_filter
                }
            }
            
        except Exception as e:
            return {
                "answer": "답변 생성 중 오류 발생",
                "documents": [],
                "metadata": {"error": str(e)}
            }
    
    def clear_memory(self):
        """대화 기록 초기화"""
        if self.memory:
            self.memory.clear()


# ============================================================================
# 9. Streamlit 연동용 초기화 함수
# ============================================================================

def initialize_rag_pipeline(vectordb_path: str = None, api_key: str = None):
    """
    Streamlit에서 사용할 수 있는 RAG 파이프라인 초기화 함수
    
    Args:
        vectordb_path: VectorDB 경로 (None이면 자동 계산)
        api_key: OpenAI API Key (None이면 환경변수 사용)
    
    Returns:
        AdvancedRAGPipeline: 초기화된 파이프라인 객체
    """
    # API Key 설정
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
    else:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError('OPENAI_API_KEY가 설정되지 않았습니다.')
    
    # LLM 및 임베딩 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        api_key=api_key
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
    
    # VectorDB 경로 설정
    if vectordb_path is None:
        vectordb_path = os.path.join(os.getcwd(), "data", "vectordb")
    
    if not os.path.exists(vectordb_path):
        raise FileNotFoundError(f"VectorDB 경로가 존재하지 않습니다: {vectordb_path}")
    
    # VectorStore 로드
    vectorstore = Chroma(
        collection_name="youth_policies",
        embedding_function=embeddings,
        persist_directory=vectordb_path
    )
    
    # 문서 로드 (BM25를 위해 필요)
    all_docs = vectorstore.get()
    
    if not all_docs or not all_docs.get('documents'):
        raise ValueError("VectorDB에 문서가 없습니다.")
    
    documents = []
    for i, doc_text in enumerate(all_docs['documents']):
        if doc_text and doc_text.strip():
            metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
            documents.append(Document(page_content=doc_text, metadata=metadata))
    
    # RAG 파이프라인 생성
    rag = AdvancedRAGPipeline(
        documents=documents,
        vectorstore=vectorstore,
        llm=llm,
        enable_router=True,
        enable_multi_query=True,
        enable_ensemble=True,
        enable_rrf=True,
        enable_memory=True,
        enable_region_filter=True,
        bm25_k=5,
        vector_k=10,
        bm25_weight=0.4,
        vector_weight=0.6
    )
    
    return rag
