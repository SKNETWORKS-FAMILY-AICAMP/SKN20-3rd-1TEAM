"""
고급 RAG 파이프라인 구현
- Router: 질문 검증 및 정제
- Multi-Query Generator: 다중 관점 쿼리 생성
- Ensemble Retriever: Dense + BM25
- RRF (Reciprocal Rank Fusion): 검색 결과 통합
- Memory Store: 대화 맥락 관리
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
import warnings

# TensorFlow 로그 억제 (dotenv 로드 전에 설정)
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
    # LangChain deprecation 경고 무시
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

# Retrievers 사용 가능 여부 확인
if not RETRIEVERS_AVAILABLE:
    print("⚠️ BM25 Retriever를 사용할 수 없습니다.")
    print("⚠️ 설치: pip install langchain-community")



# ============================================================================
# 1. Router: 질문 검증 및 정제
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
            print(f"🔀 Router: {result['category']} | Valid: {result['is_valid']}")
            
            return result
        except Exception as e:
            print(f"❌ Router Error: {e}")
            return {
                "is_valid": True,
                "category": "일반질문",
                "refined_query": query,
                "reason": "파싱 실패로 원본 사용"
            }


# ============================================================================
# 2. Multi-Query Generator: 다중 관점 쿼리 생성
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
            2.  **정책 키워드(Policy Keyword): 질문의 **핵심 의도**와 관련된 정책 키워드를 추출하여 관련된 정책만 반환할 것.(월세 -> 월세(유의어 예: 임대료 등 포함) 관련 정책만 반환)
            1.  **원본 쿼리의 핵심 키워드**를 살린 가장 직관적인 쿼리
            2.  **유사한 의미 또는 관련 정책명**을 포함하는 쿼리 (동의어 활용)
            3.  **질문의 목적**을 명확히 드러내는 문장형 쿼리

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
            
            print(f"🔍 Multi-Query 생성: {len(all_queries)}개")
            for i, q in enumerate(all_queries, 1):
                print(f"  {i}. {q}")
            
            return all_queries
        except Exception as e:
            print(f"❌ Multi-Query Error: {e}")
            return [query]


# ============================================================================
# 3. Ensemble Retriever: 다중 검색 전략
# ============================================================================

class EnsembleRetriever:
    """Dense, BM25 검색을 결합한 앙상블 리트리버"""
    
    def __init__(
        self, 
        documents: List[any],
        vectorstore: Chroma,
        bm25_k: int = 5,
        vector_k: int = 10,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6
    ):
        self.documents = documents
        self.vectorstore = vectorstore
        
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
            print("⚠️ BM25Retriever를 사용할 수 없습니다.")
            self.bm25_retriever = None
            return
        
        if not self.documents:
            print("⚠️ BM25: 문서가 없어 초기화를 건너뜁니다.")
            self.bm25_retriever = None
            return
        
        try:
            # BM25Retriever 초기화 (from_documents 사용)
            self.bm25_retriever = BM25Retriever.from_documents(
                documents=self.documents,
                k=self.bm25_k
            )
            print(f"✅ BM25 Retriever 초기화 완료 (k={self.bm25_k})")
        except TypeError as e:
            # from_documents가 실패하면 직접 초기화 시도
            try:
                self.bm25_retriever = BM25Retriever(docs=self.documents)
                self.bm25_retriever.k = self.bm25_k
                print(f"✅ BM25 Retriever 초기화 완료 (대체 방식, k={self.bm25_k})")
            except Exception as e2:
                print(f"❌ BM25 Retriever 초기화 실패: {e2}")
                self.bm25_retriever = None
        except Exception as e:
            print(f"❌ BM25 Retriever 초기화 실패: {e}")
            self.bm25_retriever = None
    
    def _build_vector(self):
        """Vector Retriever 생성"""
        try:
            # VectorStore 상태 확인
            test_search = self.vectorstore.similarity_search("테스트", k=1)
            print(f"🧪 VectorStore 테스트 검색: {len(test_search)}개 문서")
            
            self.vector_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.vector_k}
            )
            print(f"✅ Vector Retriever 초기화 완료 (k={self.vector_k})")
        except Exception as e:
            print(f"❌ Vector Retriever 초기화 실패: {e}")
            self.vector_retriever = None
    
    def dense_search(self, query: str) -> List[Tuple[any, float]]:
        """Dense 검색 (임베딩 기반)"""
        try:
            if self.vector_retriever:
                docs = self.vector_retriever.invoke(query)
                # 스코어와 함께 반환 (스코어는 1.0으로 가정)
                results = [(doc, 1.0) for doc in docs]
                print(f"  📊 Dense: {len(results)}개 문서")
                return results
            return []
        except Exception as e:
            print(f"❌ Dense Search Error: {e}")
            return []
    
    def bm25_search(self, query: str) -> List[Tuple[any, float]]:
        """BM25 검색 (키워드 기반)"""
        try:
            if self.bm25_retriever:
                docs = self.bm25_retriever.invoke(query)
                results = [(doc, 1.0) for doc in docs]
                print(f"  📊 BM25: {len(results)}개 문서")
                return results
            return []
        except Exception as e:
            print(f"❌ BM25 Search Error: {e}")
            return []
    
    def retrieve(self, queries: List[str]) -> Dict[str, List[Tuple[any, float]]]:
        """모든 검색 전략 실행"""
        all_results = {
            'dense': [],
            'bm25': []
        }
        
        for query in queries:
            print(f"🔎 검색 중: {query}")
            all_results['dense'].extend(self.dense_search(query))
            all_results['bm25'].extend(self.bm25_search(query))
        
        return all_results
    
    def get_ensemble(self, query: str) -> List[any]:
        """Ensemble 검색 (가중치 적용)"""
        if not RETRIEVERS_AVAILABLE or EnsembleRetriever is None:
            print("⚠️ EnsembleRetriever를 사용할 수 없습니다. Vector 검색만 사용합니다.")
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
                print("❌ 사용 가능한 retriever가 없습니다")
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
            print(f"🔗 Ensemble: {len(docs)}개 문서")
            return docs
            
        except Exception as e:
            print(f"❌ Ensemble Search Error: {e}")
            return []


# ============================================================================
# 4. RRF (Reciprocal Rank Fusion): 검색 결과 통합
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
        
        print(f"🔗 RRF: {len(doc_scores)}개 문서 → {len(final_docs)}개 선택")
        return final_docs


# ============================================================================
# 5. Memory Store: 대화 맥락 관리
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
# 7. Advanced RAG Pipeline: 전체 파이프라인 통합
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
        self.ensemble = EnsembleRetriever(
            documents=documents,
            vectorstore=vectorstore,
            bm25_k=bm25_k,
            vector_k=vector_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        ) if enable_ensemble else None
        self.rrf = ReciprocalRankFusion() if enable_rrf else None
        self.memory = ConversationMemory() if enable_memory else None
        
        # 최종 답변 생성 프롬프트
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 청년정책 전담 챗봇 입니다.

        역할:
        - 청년 정책(특히 주거·월세·일자리·복지) 정보를, 사용자가 이해하기 쉽게 정리해 주는 상담사입니다.
        - 반드시 '검색된 정책 정보(documents)' 안에 있는 내용만 사용해서 답변합니다.
        - 추측하거나 지어내지 말고, 문서에 없는 정보는 "문서에 없는 내용으로 당사 홈페이지 확인이 필요합니다."고 말합니다.

        출력 형식(꼭 지켜야 함):

        1. 항상 아래 문구로 시작한다.
        안녕하세요. 청년·1인 가구 생활복지·지원 챗봇입니다.

        2. 그 다음 줄에 질문을 그대로 보여준다.
        사용자 질문 : {query}

        3. 그 다음에 '답변 :'을 쓰고, 분야 → 세부 분야(2단계) → 정책 순서로 정리한다.

        3-1. 먼저 '분야(1depth)'를 묶어서 보여준다.
                - 예: 주거·월세, 일자리·취업, 창업·교육, 금융·대출, 복지·생활 등
                - 분야 이름은 문서의 대분류/중분류, 정책명, 설명 등을 참고해서 최대한 자연스럽게 정한다.

        3-2. 각 분야 안에서 '세부 분야(2depth)'를 한 번 더 나누어 보여준다.
                - 예: 
                - [분야] 주거·월세
                    - (세부) 월세 직접지원
                    - (세부) 전세자금 이자지원
                    - (세부) 공공임대·청년주택 등
                - 세부 분야 이름도 문서 내용과 키워드를 참고해서 사람이 이해하기 쉬운 단위로 정리한다.

        3-3. 각 '세부 분야(2depth)' 안에서, 해당되는 정책들을 번호를 매겨 설명한다.
                - 최소 1개, 최대 5개까지만 가장 핵심적인 정책을 골라 설명한다.

        예시 형식(구조 예시, 문구는 상황에 맞게 자연스럽게 바꿔도 됨):

        답변 :

        1. 주거·월세 지원

            1-1. 월세 직접지원
            
                [주체/지역]
                - 주체와 지역 정리

                [내용 요약]
                - 이 정책이 어떤 상황의 청년에게 무엇을 도와주는지 한두 문장으로 정리

                [특징]
                - 다른 정책과 비교했을 때의 특징, 장점·주의점 등 (문서에 있는 범위 내에서만)

                [대상]
                - 연령 : ...
                - 주거 : ...
                - 소득 : ...
                - 기타 조건 : ...


                [지원 금액·기간]
                - 월 지원 금액 : ...
                - 지원 기간 : ...


                [신청 방법]
                - 어디서 신청 : (예: 시·군청 청년정책 담당부서, 온라인 신청 등)
                - 어떻게 신청 : (예: 온라인 신청, 방문 신청, 필요 서류 등)
                - 신청 사이트(신청URL) : …
                - 참고 사이트(참고URL) : …


                [유의사항]
                
                - 시·군별 공고 시기, 세부 조건은 반드시 해당 지자체 공고문 참고
                - 중복 지원 가능 여부 등은 실제 공고 기준으로 달라질 수 있음

                4. 마지막에 '출처' 블록을 적는다.
                    - 문서 메타데이터(파일명, 페이지 정보 등)가 있으면 최대한 활용해서 작성한다.
                    - 예시:
                    출처:
                    - 2025년 지자체 청년정책 시행계획-3(전북·전남·경북·경남·제주).pdf (465페이지, 470페이지 인근)
                    - 온통청년 누리집 청년주거·월세 지원 관련 정책 항목

                5. URL 사용 방법
                    - [검색된 정책 정보] 안에 [URL 정보] 블록이 있으면,
                        그 안의 URL들을 이용해 '신청 사이트(신청URL)', '참고 사이트(참고URL)' 항목을 채운다.
                    - URL 정보가 없으면
                        "문서에 없는 내용으로 당사 홈페이지 확인이 필요합니다."
                        라고 적는다.
        
        작성 시 유의사항:
        - 정책명이 같은 것을 중복해서 쓰지 말 것.
        - 질문에서 언급한 지역(예: 경북, 대구 등)과 직접적으로 관련 있는 정책을 최우선으로 선택할 것.
        - 질문에서 '월세', '보증금', '전세' 등 키워드가 나오면, 주거·월세 관련 정책 위주로 정리할 것.
        - 숫자(지원 금액, 기간, 연령)는 가능한 한 구체적인 값으로 써 줄 것.
        
    """
    
    ),
    (
        "user",
        """
[대화 맥락]
{context}

[검색된 정책 정보]
{documents}

[현재 질문]
{query}""")
        ])
        
        self.summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
당신은 청년 정책 상담 답변을 짧게 요약하는 보조 도우미입니다.

목표:
- 사용자의 이해를 돕기 위해, 위에서 생성된 긴 답변을 핵심만 한 번 더 정리합니다.
- 정책명, 대상(누가 받을 수 있는지), 지원 유형/금액 정도만 담아 한두 단락으로 요약합니다.
- 새로운 정보를 만들지 말고, 이미 주어진 답변 내용만 재구성하세요.
- 별표(*), 샵(#), 대시(-), 백틱(`), 언더바(_) 등의 모든 특수기호 사용 금지
"""
    ),
    (
        "user",
        """
        1. 마크다운 특수문자 사용 금지
        - 별표(*), 샵(#), 대시(-), 백틱(`), 언더바(_) 등의 모든 특수기호 사용 금지
        2. 다음 답변을 사용자가 빠르게 이해할 수 있도록 핵심만 요약해줘.
        
[전체 답변]
{answer}

        
"""
    ),
])

    
    def query(self, user_query: str) -> Dict:
        """전체 파이프라인 실행"""
        print(f"\n{'='*60}")
        print(f"📝 사용자 질문: {user_query}")
        print(f"{'='*60}")
        
        # 1. Router: 질문 검증 및 정제
        if self.router:
            route_result = self.router.route(user_query)
            if not route_result['is_valid']:
                return {
                    "answer": "죄송합니다. 질문을 이해할 수 없습니다. 다시 한번 말씀해 주세요.",
                    "documents": [],
                    "metadata": route_result
                }
            query = route_result['refined_query']
        else:
            query = user_query
        
        # 2. Multi-Query: 다중 쿼리 생성
        if self.multi_query:
            queries = self.multi_query.generate(query)
        else:
            queries = [query]
        
        # 3. Ensemble Retriever: 다중 검색
        if self.ensemble:
            search_results = self.ensemble.retrieve(queries)
        else:
            search_results = {'dense': self.vectorstore.similarity_search_with_score(query, k=5)}
        
        # 4. RRF: 검색 결과 통합 (top_k 증가)
        if self.rrf:
            docs = self.rrf.fuse(search_results, top_k=20)
        else:
            docs = [doc for doc, score in search_results['dense']]
        
        # 5. Memory: 대화 맥락 가져오기
        if self.memory:
            context = self.memory.get_context()
        else:
            context = "이전 대화 없음"
        
        
        # 6. LLM: 최종 답변 생성
                # 6. LLM: 최종 답변 생성
        doc_text_list = []

        for i, doc in enumerate(docs[:10]):
            md = doc.metadata or {}

            # 정책명 (메타데이터 키 여러 개 가능성 고려)
            policy_name = (
                md.get("정책명")
                or md.get("policy_name")
                or md.get("plcyNm")
                or f"정책 {i+1}"
            )

            # ✅ URL 메타데이터 (네 JSON 키 그대로 사용)
            apply_url = md.get("신청URL")
            ref_url1 = md.get("참고URL1")

            url_lines = []

            if apply_url:
                url_lines.append(f"- 신청URL: {apply_url}")
            else:
                url_lines.append("- 신청URL: 문서에 없는 내용으로 당사 홈페이지 확인이 필요합니다.")

            if ref_url1:
                url_lines.append(f"- 참고URL1: {ref_url1}")

            url_block = "\n".join(url_lines)

            # 이 한 덩어리가 LLM에 전달될 "한 정책 블록"
            one_doc_text = (
                f"[정책 {i+1}] {policy_name}\n"
                f"{doc.page_content[:800]}\n"   # 필요하면 500/800 숫자 조절
                f"[URL 정보]\n"
                f"{url_block}"
            )

            doc_text_list.append(one_doc_text)

        # 🔹 최종적으로 프롬프트에 들어가는 documents 문자열
        docs_text = "\n\n".join(doc_text_list)

        
        try:
            response = self.answer_prompt | self.llm | StrOutputParser()
            answer = response.invoke({
                "context": context,
                "documents": docs_text,
                "query": user_query
            })
            
            # 7. 요약 생성 (Chain of Thought)
            summary_response = self.summary_prompt | self.llm | StrOutputParser()
            summary = summary_response.invoke({"answer": answer})
            
            # 메모리에 저장
            if self.memory:
                self.memory.add_message("user", user_query)
                self.memory.add_message("assistant", answer)
            
            print(f"\n✅ 답변 생성 완료")
            print(f"📌 요약 생성 완료")
            print(f"{'='*60}\n")
            
            return {
                "answer": answer,
                "summary": summary,
                "documents": docs,
                "metadata": {
                    "queries": queries,
                    "num_docs_retrieved": len(docs),
                    "has_context": bool(self.memory and self.memory.messages)
                }
            }
            
        except Exception as e:
            print(f"❌ Answer Generation Error: {e}")
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "documents": [],
                "metadata": {"error": str(e)}
            }
    
    def clear_memory(self):
        """대화 기록 초기화"""
        if self.memory:
            self.memory.clear()
            print("🗑️ 메모리 초기화 완료")


# ============================================================================
# 8. Streamlit 연동을 위한 초기화 함수
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
        temperature=0.7,
        api_key=api_key
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
    
    # VectorDB 경로 설정
    if vectordb_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        vectordb_path = os.path.join(project_root, "data", "vectordb")
    
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
        bm25_k=5,
        vector_k=10,
        bm25_weight=0.4,
        vector_weight=0.6
    )
    
    return rag


# ============================================================================
# 8. 사용 예시
# ============================================================================

def main():
    """고급 RAG 파이프라인 테스트"""
    
    # 환경 변수 확인
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY가 설정되지 않았습니다.')
    
    # LLM 및 임베딩 초기화
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        api_key=api_key
    )
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )
    
    # VectorDB 로드 (프로젝트 루트 기준 경로)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    vectordb_path = os.path.join(project_root, "data", "vectordb")
    
    print(f"📂 VectorDB 경로: {vectordb_path}")
    print(f"📂 경로 존재 여부: {os.path.exists(vectordb_path)}")
    
    if not os.path.exists(vectordb_path):
        print("❌ VectorDB 경로가 존재하지 않습니다. build_vectordb.py를 먼저 실행하세요.")
        return
    
    vectorstore = Chroma(
        collection_name="youth_policies",
        embedding_function=embeddings,
        persist_directory=vectordb_path
    )
    
    # 문서 로드 (BM25를 위해 필요)
    # ChromaDB에서 모든 문서 가져오기
    all_docs = vectorstore.get()
    print(f"📊 ChromaDB 로드 결과: {len(all_docs.get('documents', []))}개 문서")
    
    if not all_docs or not all_docs.get('documents'):
        print("❌ VectorDB에 문서가 없습니다. build_vectordb.py를 먼저 실행하세요.")
        return
    
    documents = []
    if all_docs and 'documents' in all_docs:
        from langchain_core.documents import Document
        for i, doc_text in enumerate(all_docs['documents']):
            if doc_text and doc_text.strip():  # 빈 문서 제외
                metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))
    
    print(f"📚 총 {len(documents)}개 문서 로드 완료")
    
    # 고급 RAG 파이프라인 생성
    rag = AdvancedRAGPipeline(
        documents=documents,
        vectorstore=vectorstore,
        llm=llm,
        enable_router=True,
        enable_multi_query=True,
        enable_ensemble=True,
        enable_rrf=True,
        enable_memory=True,
        bm25_k=5,
        vector_k=10,
        bm25_weight=0.4,
        vector_weight=0.6
    )
    
    # 테스트 질의
    queries = [
        "청년주택 정책",
    ]
    
    for query in queries:
            result = rag.query(query)
            print(f"\n질문: {query}")
            print(f"\n📄전체 답변:\n{result['answer']}")
            if 'summary' in result:
                print(f"\n📌요약:\n{result['summary']}")
            print(f"\n문서 수: {result['metadata'].get('num_docs_retrieved', 0)}")
            print("-" * 60)


if __name__ == "__main__":
    main()
        

