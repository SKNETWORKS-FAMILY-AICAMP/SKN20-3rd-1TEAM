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
            ("system", """당신은 검색 쿼리를 다양한 관점으로 확장하는 전문가입니다.

주어진 질문을 3가지 다른 관점에서 재구성하세요:
1. 키워드 중심 쿼리
2. 의미 중심 쿼리
3. 맥락 중심 쿼리

{region_instruction}

            각 쿼리는 한 줄로 작성하고, 번호 없이 줄바꿈으로 구분하세요."""),
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
            ("system", """당신은 청년 정책 전문 상담사입니다.

            검색된 정책 정보와 대화 맥락을 바탕으로 친절하고 정확한 답변을 제공하세요.

답변 원칙:
1. 검색된 문서 정보를 기반으로 답변
2. 정책명, 신청 기간, 지원 내용 등 구체적으로 설명
3. 대화 맥락을 고려하여 자연스럽게 답변
4. 정보가 부족하면 솔직하게 말하기
5. **제공된 모든 정책을 가능한 포함하여 답변하세요** (최소 3개 이상)"""),
            ("user", """[대화 맥락]
{context}

[사용자 프로필]
{profile}

            [검색된 정책 정보]
            {documents}

[현재 질문]
{query}""")
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
        docs_text = "\n\n".join([
            f"[정책 {i+1}] {doc.metadata.get('policy_name', '제목 없음')}\n{doc.page_content[:500]}"
            for i, doc in enumerate(docs[:10])
        ])
        
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
        "대구 월세 지원",
    ]
    
    for query in queries:
        result = rag.query(query)
        print(f"\n질문: {query}")
        print(f"\n📄 전체 답변:\n{result['answer']}")
        print(f"\n📌 요약:\n{result['summary']}")
        print(f"\n문서 수: {result['metadata'].get('num_docs_retrieved', 0)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
