"""
청년 정책 RAG Pipeline - RRF 확장 (검색단계만 RRF 적용)
Reciprocal Rank Fusion (RRF)을 이용한 Dense + Sparse + Statistical 3-way 하이브리드 검색
"""

from collections import defaultdict
from typing import List, Optional, Dict, Any

from youth_policy_rag import YouthPolicyRAG, safe_print  # 원본 클래스 및 도구 재사용
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import TFIDFRetriever


class YouthPolicyRAG_RRF(YouthPolicyRAG):
    """
    RRF(Reciprocal Rank Fusion)를 활용한 RAG 시스템 확장
    - Dense retriever (벡터 유사도) + BM25 (키워드 기반) + TF-IDF (통계 기반) 3-way 앙상블
    """
    
    def __init__(self, *args,
                 top_k: int = 10, # 최종 반환할 상위 문서 수
                 rrf_k: int = 60, # RRF 계산 시 보정 상수
                 **kwargs): 
        super().__init__(*args, **kwargs)
        self.top_k: int = top_k
        self.rrf_k: int = rrf_k

        # TF-IDF retriever 추가
        try:
            self.tfidf_retriever = TFIDFRetriever.from_documents(self.documents)
            self.tfidf_retriever.k = self.top_k
            safe_print("✅ TF-IDF Retriever 준비 완료")
        except Exception as e:
            safe_print(f"⚠️ TF-IDF retriever 초기화 실패: {e}", force=True)
            self.tfidf_retriever = None
        
         # BM25가 있다면 k 동기화
        try :
            if hasattr(self, "bm25_retriever") and self.bm25_retriever is not None:
                 self.bm25_retriever.k = self.top_k
        except Exception:
            pass

   
    # 검색 유틸리티 (RRF 기반)
    def _get_doc_identifier(self, doc:Any) -> str:
        """
        문서 고유 식별 키 생성
        메타데이터의 정책명 우선 사용, 없으면 콘텐츠 기반 생성
        """
        if getattr(doc, "metadata", None):
            policy_name = doc.metadata.get("정책명", "")
            if policy_name:
                return f"policy_{policy_name}"
        
        # fallback: 콘텐츠 + 메타데이터 조합
        meta = doc.metadata or {}
        identifier = (
            str(meta.get("정책명", "unknown")),
            str(meta.get("주관기관명", "")),
            doc.page_content[:100]  # 더 긴 콘텐츠 사용으로 충돌 감소
        )
        return str(hash(identifier))
    
    def _dense_retrieve(self, question: str) -> List:
        """
        Dense retriever (벡터 유사도 기반)로 문서 검색
        """
        docs = []
        try:
            if hasattr(self, "vectorstore") and self.vectorstore is not None:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
                docs = retriever.invoke(question)
                safe_print(f"  📊 Dense 검색: {len(docs)}개 문서 반환")
        except Exception as e:
            safe_print(f"⚠️ Dense 검색 실패: {e}", force=True)
        return docs or []

    def _sparse_retrieve(self, question: str) -> List:
        """
        Sparse retriever (BM25 키워드 기반)로 문서 검색
        부모 클래스의 BM25 retriever 재사용
        """
        docs = []
        try:
            if hasattr(self, "bm25_retriever") and self.bm25_retriever is not None:
                docs = self.bm25_retriever.invoke(question)
                safe_print(f"  🔤 BM25 검색: {len(docs)}개 문서 반환")
            else:
                safe_print("⚠️ BM25 retriever가 초기화되지 않았습니다.")
        except Exception as e:
            safe_print(f"⚠️ Sparse (BM25) 검색 실패: {e}", force=True)
        return docs or []
    
    def _statistical_retrieve(self, question: str) -> List:
        """
        Statistical retriever (TF-IDF 기반)로 문서 검색
        """
        docs = []
        try:
            if self.tfidf_retriever is not None:
                docs = self.tfidf_retriever.invoke(question)
                safe_print(f"  📈 TF-IDF 검색: {len(docs)}개 문서 반환")
            else:
                safe_print("⚠️ TF-IDF retriever가 초기화되지 않았습니다.")
        except Exception as e:
            safe_print(f"⚠️ Statistical (TF-IDF) 검색 실패: {e}", force=True)
        return docs or []

    def reciprocal_rank_fusion(self, ranked_lists: List[List], k: Optional[int] = None,
                               top_n: Optional[int] = None) -> List:
        """
        RRF (Reciprocal Rank Fusion) 알고리즘
        여러 retriever의 순위 기반 검색 결과를 통합
        
        Args:
            ranked_lists: 각 retriever의 순위 문서 리스트 [[doc1, doc2, ...], ...]
            k: RRF 보정 상수
            top_n: 최종 반환 문서 수
            
        Returns:
            RRF 점수 기반으로 정렬된 상위 top_n개 문서 리스트
        """
        if k is None:
            k = self.rrf_k
        if top_n is None:
            top_n = self.top_k

        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Any] = {}

        # 각 retriever의 결과에서 RRF 점수 합산
        for rank_idx, docs in enumerate(ranked_lists):
            retriever_name = ["Dense", "BM25", "TF-IDF"][rank_idx] if rank_idx < 3 else f"Retriever_{rank_idx}"
            safe_print(f"  처리 중: {retriever_name} ({len(docs)}개)")
            
            for rank, doc in enumerate(docs):
                doc_id = self._get_doc_identifier(doc)
                doc_map[doc_id] = doc
                rrf_score = 1.0 / (k + rank + 1)
                scores[doc_id] += rrf_score

        # 점수가 높은 순으로 정렬 후 상위 top_n개만 반환
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        result_docs = [doc_map[doc_id] for doc_id in sorted_ids[:top_n]]
        
        safe_print(f"  ✅ RRF 결과: {len(result_docs)}개 문서 (상위 {top_n}개)")
        return result_docs

    def _retrieve_with_rrf(self, question: str) -> List:
        """
        RRF 기반 3-way 하이브리드 검색 실행
        Dense + BM25 + TF-IDF 결과를 RRF로 통합
        
        Args:
            question: 사용자 질문
            
        Returns:
            RRF 통합 점수 기반 상위 문서 리스트
        """
        safe_print(f"🔎 RRF 검색 시작 (top_k={self.top_k}, rrf_k={self.rrf_k})")
        
        dense_docs = self._dense_retrieve(question)
        sparse_docs = self._sparse_retrieve(question)
        statistical_docs = self._statistical_retrieve(question)

        # 모두 비어있으면 빈 리스트 반환
        if not any([dense_docs, sparse_docs, statistical_docs]):
            safe_print("⚠️ 모든 retriever에서 결과가 없습니다.")
            return []

        # 하나라도 있으면 RRF 수행 (있는 것만)
        ranked_lists = []
        if dense_docs:
            ranked_lists.append(dense_docs)
        if sparse_docs:
            ranked_lists.append(sparse_docs)
        if statistical_docs:
            ranked_lists.append(statistical_docs)

        return self.reciprocal_rank_fusion(ranked_lists)

    
    # 오버라이드: query() - RRF 기반 검색 활용

    def query(self, question: str): 

        # 사용자 정보 표시용 문자열
        user_info = ""
        if self.user_age or self.user_region:
            age_str = f"{self.user_age}세" if self.user_age is not None else ""
            region_str = f"{self.user_region}" if self.user_region else ""
            # 빈 문자열을 제외하고 조합
            parts = [p for p in [age_str, region_str] if p]
            if parts:
                user_info = f" ({', '.join(parts)})"
        safe_print(f"\n🔍 질문: {question}{user_info}")

        # 질문 라우팅 (원본 메서드 사용)
        routing_result = self.route_query(question)
        action = routing_result.get('action')

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
            safe_print("⏳ RRF 정책 검색 중...\n")

            # ✨ RRF를 이용해 Dense + BM25 + TF-IDF 3-way 앙상블 검색 수행
            docs = self._retrieve_with_rrf(question)
            context = self._format_docs(docs)
            chat_history_txt = self._format_chat_history()

            prompt = ChatPromptTemplate.from_template(
                """당신은 청년 정책 전문 상담사입니다.
            아래는 지금까지의 대화 기록과, 검색된 정책 정보입니다.

            [대화 기록]
            {chat_history}

            [정책 정보]
            {context}

            [사용자 질문]
            {question}

            답변 가이드라인:
            1. 검색된 모든 정책을 빠짐없이 소개하세요.
            2. 각 정책마다 다음 정보를 원본 그대로 포함하세요:
               - 정책명
               - 담당기관
               - 지원대상 (연령, 학력, 거주지)
               - 지원내용 (구체적인 금액, 지원 방식)
               - 신청기간
               - 참고링크
            3. 정보를 요약하거나 생략하지 말고 제공된 정보를 그대로 전달하세요.
            4. 정책 정보에 없는 내용은 추가하지 마세요.
            5. 친근하고 격려하는 톤으로 작성하되, 정보는 정확하고 상세하게 제공하세요.
            6. 각 정책 사이에 구분선(━━━)을 넣어 읽기 쉽게 하세요.
            7. 연령이 0세 ~ 0세인 경우 "제한없음"으로 표현하세요.
            8. 연령이 n세 ~ 0세인 경우 "n세 이상"으로 표현하세요.
            답변:"""
            )

            raw_answer = (prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": chat_history_txt,
                 "context": context,
                 "question": question})

            # Self-RAG 검증은 원본 메서드로 수행 (근거 기반 확인)
            answer = self.self_rag_verify(question, raw_answer, context)

        # 대화 기록에 저장 (원본과 동일)
        if self.chat_history is not None and answer:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

        return answer



# 테스트 및 사용 예시

def main():
    """RRF RAG 시스템 테스트"""
    print("=" * 70)
    print("🚀 청년 정책 RAG (RRF 확장) 테스트")
    print("=" * 70)
    
    try:
        # RRF 시스템 초기화
        rag = YouthPolicyRAG_RRF(use_multi_query=True, top_k=10, rrf_k=60)
        
        # 테스트 쿼리
        test_queries = [
            "창업 지원금 정책이 있나요?",
            "청년 취업 지원 프로그램",
        ]
        
        # 사용자 정보 설정 (선택)
        rag.set_user_info(age=27, region="서울특별시", education="대학교 졸업")
        
        for query in test_queries:
            print(f"\n📝 질문: {query}")
            print("-" * 70)
            answer = rag.query(query)
            print(f"\n🤖 답변:\n{answer}\n")
            print("=" * 70)
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
