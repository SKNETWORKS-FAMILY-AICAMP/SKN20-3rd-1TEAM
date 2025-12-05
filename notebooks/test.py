"""
청년 정책 RAG Pipeline
단계별로 구축하는 고급 RAG 시스템
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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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
                print(f"⚠️ Retriever 오류: {e}")
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
        template = """당신은 AI 검색 전문가입니다. 사용자의 질문을 다양한 관점에서 재작성하여 더 나은 검색 결과를 얻으려고 합니다.

원본 질문: {question}

위 질문을 **3가지 다른 방식**으로 재작성하세요:
1. 더 구체적으로
2. 더 넓은 관점에서
3. 다른 키워드 사용

응답 형식 (JSON):
{{
  "queries": [
    "재작성된 질문 1",
    "재작성된 질문 2",
    "재작성된 질문 3"
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
            queries = result.get("queries", [question])
            
            print(f"🔄 MultiQuery 생성: {len(queries)}개")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
            
            return queries
            
        except Exception as e:
            print(f"⚠️ MultiQuery 생성 실패: {e}, 원본 질문만 사용")
            return [question]


class YouthPolicyRAG:
    """청년 정책 RAG 시스템"""
    
    def __init__(self, db_path="../data/vectordb"):
        """
        초기화
        
        Args:
            db_path: ChromaDB 경로
        """
        print("🚀 RAG Pipeline 초기화 중...")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        
        # 임베딩 모델
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        
        # Vector Store 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_db_path = os.path.join(current_dir, db_path)
        
        self.vectorstore = Chroma(
            persist_directory=full_db_path,
            collection_name="youth_policies",
            embedding_function=self.embeddings
        )
        
        # ChromaDB collection 직접 접근 (필터링용)
        chroma_client = chromadb.PersistentClient(path=full_db_path)
        self.collection = chroma_client.get_collection(name="youth_policies")
        
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
        
        # 사용자 정보 (나이, 지역)
        self.user_age = None
        self.user_region = None
        
        # MultiQuery 사용 여부 (기본: True)
        self.use_multi_query = True
        
        # 프롬프트 템플릿
        self.prompt = self._create_prompt()
        
        # Router 프롬프트
        self.router_prompt = self._create_router_prompt()
        
        # RAG 체인 구성
        self.rag_chain = self._build_chain()

        self.chat_history = []      # 대화 메모리용 리스트
        self.self_rag_prompt = self._create_self_rag_prompt()  # Self-RAG 프롬프트
        
        
        print("✅ RAG Pipeline 초기화 완료!")
    
    def _load_documents(self):
        """ChromaDB에서 문서 로딩 (한 번만 수행)"""
        print("📄 문서 로딩 중...")
        all_data = self.collection.get()
        
        documents = []
        for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
            documents.append(Document(
                page_content=doc_text,
                metadata=metadata
            ))
        
        print(f"✅ 문서 로딩 완료 (문서 수: {len(documents)}개)")
        return documents
    
    def _init_bm25_retriever(self):
        """BM25 Retriever 초기화 (키워드 기반 검색)"""
        print("📚 BM25 Retriever 초기화 중...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 10  # 상위 10개 검색
        print("✅ BM25 Retriever 초기화 완료")
    
    def _init_tfidf_retriever(self):
        """TF-IDF Retriever 초기화 (통계 기반 검색)"""
        print("📊 TF-IDF Retriever 초기화 중...")
        self.tfidf_retriever = TFIDFRetriever.from_documents(self.documents)
        self.tfidf_retriever.k = 10  # 상위 10개 검색
        print("✅ TF-IDF Retriever 초기화 완료")
    
    def _init_ensemble_retriever(self):
        """Ensemble Retriever 초기화 (Dense + BM25 + TF-IDF 3-way hybrid)"""
        print("🔗 Ensemble Retriever 생성 중 (3-way hybrid)...")
        
        # Dense Vector Retriever (의미 기반) - 유사도 점수 포함
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.3  # 유사도 30% 이상만 반환
            }
        )
        
        # 3-way Hybrid: Dense + BM25 + TF-IDF (직접 구현)
        self.ensemble_retriever = SimpleEnsembleRetriever(
            retrievers=[vector_retriever, self.bm25_retriever, self.tfidf_retriever],
            weights=[0.5, 0.3, 0.2]  # Dense 50%, BM25 30%, TF-IDF 20%
        )
        print("✅ Ensemble Retriever 생성 완료 (Dense + BM25 + TF-IDF)")
        print("   가중치: Dense 50% | BM25 30% | TF-IDF 20%")
    
    def _create_router_prompt(self):
        """Router 프롬프트 생성"""
        template = """당신은 질문을 분석하여 적절한 작업을 선택하는 라우터입니다.

질문: {question}

다음 중 하나를 선택하세요:

1. SEARCH_POLICY
   - 청년 정책 검색이 필요한 경우
   - 예: "창업 지원금", "취업 지원", "주거 지원", "대출", "교육" 등

2. GENERAL_CHAT
   - 일반적인 인사, 감사 표현
   - 예: "안녕하세요", "고맙습니다", "도움이 되었어요"

3. REQUEST_INFO
   - 사용자 정보(나이, 지역)가 필요한 경우
   - 예: 정책 질문인데 사용자 정보가 없는 경우

4. CLARIFY
   - 질문이 불명확하여 추가 정보가 필요한 경우
   - 예: "정책", "지원금" 같이 너무 광범위한 질문

**중요**: 반드시 JSON 형식으로만 답변하세요.

응답 형식:
{{
  "action": "SEARCH_POLICY",
  "reason": "창업 지원금 관련 정책 검색 필요",
  "keywords": ["창업", "지원금"]
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def _create_prompt(self):
        """프롬프트 템플릿 생성"""
        template = """당신은 청년 정책 전문 상담사입니다.
사용자의 질문에 대해 제공된 정책 정보를 바탕으로 친절하고 정확하게 답변하세요.

📋 정책 정보:
{context}

❓ 사용자 질문:
{question}

💡 답변 가이드라인:
1. 제공된 정책 정보만 사용하세요
2. 정책명, 지원내용, 신청방법을 명확히 설명하세요
3. 정보가 부족하면 "제공된 정보에는 없습니다"라고 말하세요
4. 친근하고 격려하는 톤으로 작성하세요
5. 필요시 추가 질문을 유도하세요
6. 정책에 관련되지 않은 질문에는 답변하지 마세요
7. 현재 날짜를 기준으로 최신 정보를 제공하세요

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

반드시 아래 JSON 형식으로만 출력하세요:

{{
  "is_grounded": true or false,
  "issues": ["문제1", "문제2"],
  "suggested_fix": "문제가 있을 경우, 더 안전하고 정확한 수정 답변을 한글로 작성"
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def self_rag_verify(self, question:str, answer:str):
        """Self-rag : 답변이 컨텍스트에 근거하는지 검증"""
        try :
            context = self._format_docs(docs)
            chain = self.self_rag_prompt | self.llm | StrOutputParser()
            resp = chain.invoke({"context": context, "answer": answer})
            # JSON만 추출
            if "```json" in resp:
                resp = resp.split("```json")[1].split("```")[0].strip()
            elif "```" in resp:
                resp = resp.split("```")[1].split("```")[0].strip()
            result = json.loads(resp)
            is_grounded = result.get("is_grounded",True)

            if is_grounded :
                print("✅ Self-RAG : 근거 기반 답변으로 판단")
                return answer
            
            # 수정 제안이 없으면 일단 원답 유지
            return answer
        except Exception as e:
            print(f"⚠️ Self-RAG 검증 실패: {e}")
            return answer

    def _build_chain(self):
        """RAG 체인 구성"""
        chain = (
            {
                "context": RunnableLambda(self._retrieve_and_filter) | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
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
                print(f"⚠️ 쿼리 '{query}' 검색 오류: {e}")
                continue
        
        print(f"🔍 총 검색 결과: {len(all_docs)}개 (중복 제거)")
        
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
                        print(f"  ✕ 종료된 정책: {policy_name} (종료일: {end_date_str})")
                else:
                    # 파싱 실패 시 포함
                    active_docs.append(doc)
            except:
                # 예외 발생 시 포함
                active_docs.append(doc)
        
        print(f"✅ 기간 필터링 후: {len(active_docs)}개 (종료된 정책 제외)")
        
        # 사용자 정보가 없으면 기간 필터링만 적용하고 반환
        if not (self.user_age or self.user_region):
            return active_docs[:5]
        
        # 나이/지역 필터링 시작
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
            
            # 지역 필터링 (계층적 매칭: 전국 → 시/도 → 시/군/구)
            region_match = True
            if self.user_region:
                org_name = metadata.get('주관기관명', '')
                additional_cond = metadata.get('추가자격조건', '')
                reg_group = metadata.get('재공기관그룹', '')
                
                policy_name = metadata.get('정책명', 'N/A')
                
                # 1순위: 전국 정책은 항상 포함
                if '중앙부처' in reg_group or '전국' in org_name:
                    region_match = True
                    print(f"  ✓ 전국 정책: {policy_name} (기관: {org_name})")
                else:
                    # 2순위: 시/도 단위 매칭 (구/군 입력 시에도 시/도 정책 포함)
                    sido_list = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                               '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
                    
                    user_sido = None
                    for sido in sido_list:
                        if sido in self.user_region:
                            user_sido = sido
                            break
                    
                    # 시/도 매칭 확인
                    if user_sido and user_sido in org_name:
                        region_match = True
                        print(f"  ✓ 시/도 매칭: {policy_name} (시/도: {user_sido}, 기관: {org_name})")
                    else:
                        # 3순위: 구/군 단위 상세 매칭
                        region_clean = self.user_region.replace('특별시', '').replace('광역시', '').replace('특별자치시', '')
                        region_clean = region_clean.replace('도', '').replace('시', '').replace('군', '').replace('구', '').strip()
                        
                        user_region_tokens = []
                        if user_sido:
                            user_region_tokens.append(user_sido)
                        
                        for token in region_clean.split():
                            if token and token not in user_region_tokens:
                                user_region_tokens.append(token)
                        
                        region_match = False
                        for token in user_region_tokens:
                            if token in org_name or token in additional_cond:
                                region_match = True
                                print(f"  ✓ 상세 매칭: {policy_name} (토큰: {token}, 기관: {org_name})")
                                break
                        
                        if not region_match:
                            print(f"  ✗ 제외: {policy_name} (기관: {org_name})")
            
            # 두 조건 모두 만족하면 포함
            if age_match and region_match:
                filtered_docs.append(doc)
        
        print(f"✅ 필터링 후: {len(filtered_docs)}개")
        
        # 결과가 너무 적으면 전국 정책만이라도 반환
        if len(filtered_docs) < 3:
            print("⚠️ 필터링 결과 부족, 전국 정책 추가 검색")
            for doc in active_docs:
                if len(filtered_docs) >= 5:
                    break
                metadata = doc.metadata
                reg_group = metadata.get('재공기관그룹', '')
                if '중앙부처' in reg_group and doc not in filtered_docs:
                    filtered_docs.append(doc)
        
        return filtered_docs[:5]
    
    def _format_docs(self, docs):
        """문서 포맷팅"""
        if not docs:
            return "검색된 정책이 없습니다."
        
        formatted = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            formatted.append(f"""
[정책 {i}]
정책명: {metadata.get('정책명', 'N/A')}
분야: {metadata.get('대분류', 'N/A')} > {metadata.get('중분류', 'N/A')}
담당기관: {metadata.get('주관기관명', 'N/A')}
연령: {metadata.get('지원최소연령', 'N/A')}세 ~ {metadata.get('지원최대연령', 'N/A')}세
지원금액: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원
내용: {doc.page_content[:500]}...
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

    def query(self, question: str):
        """
        질문에 답변 (Router 적용)
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 답변
        """
        user_info = ""
        if self.user_age or self.user_region:
            user_info = f" (나이: {self.user_age}세, 지역: {self.user_region})"
        
        print(f"\n🔍 질문: {question}{user_info}")
        
        # 1단계: Router로 질문 분석
        routing_result = self.route_query(question)
        action = routing_result.get('action')
        
        # 2단계: Action에 따라 처리
        if action == "GENERAL_CHAT":
            # 일반 대화 - 검색 없이 직접 응답
            print("💬 일반 대화 모드\n")
            chat_prompt = ChatPromptTemplate.from_template(
                "당신은 친근한 청년 정책 상담사입니다. 다음 질문에 간단히 답변하세요.\n\n질문: {question}\n\n답변:"
            )
            response = (chat_prompt | self.llm | StrOutputParser()).invoke({"question": question})
            return response
        
        elif action == "REQUEST_INFO":
            # 사용자 정보 요청
            print("📝 사용자 정보 필요\n")
            return """더 정확한 정책을 추천해드리기 위해 정보가 필요합니다! 😊

다음 정보를 알려주시겠어요?
1. 나이: 만 몇 세이신가요?
2. 지역: 어디에 거주하시나요? (예: 서울특별시, 경기도 의정부시)

정보를 입력하시면 맞춤형 정책을 찾아드리겠습니다!"""
        
        elif action == "CLARIFY":
            # 질문 명확화 요청
            print("❓ 질문 명확화 필요\n")
            return """질문을 좀 더 구체적으로 말씀해주시겠어요? 😊

예를 들면:
- "창업 지원금이 궁금해요"
- "청년 취업 지원 프로그램 알려주세요"
- "전월세 대출 정책이 있나요?"

구체적인 분야를 말씀해주시면 더 정확한 정책을 찾아드릴게요!"""
        
        else:  # SEARCH_POLICY
            # 정책 검색 - RAG 체인 실행
            print("⏳ 정책 검색 중...\n")
            response = self.rag_chain.invoke(question)
            return response
    
    def set_user_info(self, age=None, region=None):
        """
        사용자 정보 설정
        
        Args:
            age: 나이
            region: 지역 (예: "경기도 의정부시")
        """
        self.user_age = age
        self.user_region = region
        
        info = []
        if age:
            info.append(f"나이 {age}세")
        if region:
            info.append(f"지역 {region}")
        
        if info:
            print(f"✅ 사용자 정보 설정: {', '.join(info)}")
            print(f"   → 전국/중앙부처 정책 + {region} 정책이 함께 검색됩니다.")
    
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
                    print(f"ℹ️  사용자 정보 이미 있음 (나이: {self.user_age}, 지역: {self.user_region})")
                    result['action'] = 'SEARCH_POLICY'
                    result['reason'] = '사용자 정보 있음, 정책 검색 진행'
            
            print(f"🎯 라우팅 결과: {result['action']} - {result.get('reason', '')}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ 라우팅 오류: {e}, 기본 검색으로 진행")
            return {
                "action": "SEARCH_POLICY",
                "reason": "라우팅 실패, 기본 검색",
                "keywords": []
            }
        
    def advanced_query(self, question:str) -> str:
        """대화 메모리 + Self-RAG 적용 고급 질의응답 함수.
        기존 query()는 건드리지 않고, 이 메서드를 별도로 사용하면 됨."""
        user_info = ""
        if self.user_age or self.user_region:
            user_info = f" (나이: {self.user_age}세, 지역: {self.user_region})"
        print(f"\n🔍 [ADV]질문: {question}{user_info}")

        # 1단계 : Router 사용 (기존 로직 재사용)
        routing_result = self.route_query(question)
        action = routing_result.get('action')
        answer = ""

        # 2단계 : Action에 따라 처리
        if action == "GENERAL_CHAT":
            print("💬 [ADV]일반 대화 모드\n")
            prompt = ChatPromptTemplate.from_template(
                """당신은 친근한 청년 정책 상담사입니다.
                아래는 지금까지의 대화 기록입니다
                
                [대화 기록]
                {chat_history}
                [사용자 질문]
                {question}

                간단하고 따뜻하게 답변하세요.

                답변:"""
                )
            chat_history_txt = self._format_chat_history()
            answer = (prompt | self.llm | StrOutputParser()).invoke(
                {"chat_history": chat_history_txt, "question": question})
        elif action == "REQUEST_INFO":
            print("📝 [ADV]사용자 정보 필요\n")
            answer = """더 정확한 정책을 추천해드리기 위해 정보가 필요합니다! 😊
            
            다음 정보를 알려주시겠어요?
            1. 나이: 만 몇 세이신가요?
            2. 지역: 어디에 거주하시나요? (예: 서울특별시, 경기도 의정부시)
            
            정보를 입력하시면 맞춤형 정책을 찾아드리겠습니다!"""
        else : # SEARCH_POLICY or 기타
            print("⏳ [ADV]정책 검색 중...\n")
            # 1) 문서 검색
            docs = self._retrieve_and_filter(question)
            # 2) 컨텍스트 포매팅
            context = self._format_docs(docs)
            # 3) 대화 기록
            chat_history_txt = self._format_chat_history()

            # 4) 1차 답변 생성 (대화 기록 + 컨텍스트 같이 제공)
            prompt = ChatPromptTemplate.from_template("""당신은 청년 정책 전문 상답사입니다
            아래는 지금까지의 대화 기록과, 검색된 정책 정보입니다.
            
            [대화 기록]
            {chat_history}
            
            [정책 정보]
            {context}

            [사용자 질문]
            {question}
            답변 가이드라인:
            1. 제공된 정책 정보만 사용하세요.
            2. 정책명, 지원내용, 신청방법을 명확히 설명하세요.
            3. 정보가 부족하면 "제공된 정보에는 없습니다"라고 말하세요.
            4. 친근하고 격려하는 톤으로 작성하세요.
            5. 필요시 추가 질문을 유도하세요.
                                                      
            답변:"""
                    )
            raw_answer = (prompt | self.llm | StrOutputParser()).invoke(                                                                                                                              
                {"chat_history": chat_history_txt,
                 "context": context,
                 "question": question})
            
            # 5) Self-RAG 검증
            answer = self._self_rag_verify(question, raw_answer, docs)
        # 3단계 : 대화 메모리에 저장
        if self.chat_history is not None and answer:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))
        return answer

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
                
                # 답변 생성
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
