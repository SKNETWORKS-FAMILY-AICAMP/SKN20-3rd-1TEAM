"""
청년 정책 Q&A 챗봇 - Streamlit Frontend
"""

import streamlit as st
import os
from advanced_rag_pipeline import initialize_rag_pipeline

# ========================================
# CSS 스타일
# ========================================

def apply_custom_css():
    """커스텀 CSS 적용"""
    st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #757575;
    }
    
    .policy-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .summary-box {
        background-color: #fff9c4;
        border-left: 4px solid #fbc02d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ========================================
# RAG 시스템 초기화
# ========================================

@st.cache_resource
def load_rag_pipeline():
    """Advanced RAG 파이프라인 초기화 (캐싱)"""
    try:
        return initialize_rag_pipeline()
    except Exception as e:
        st.error(f"❌ RAG 파이프라인 초기화 실패: {e}")
        return None


# ========================================
# UI 렌더링 함수
# ========================================

def render_question_interface(rag):
    """질문 입력 및 답변 인터페이스"""
    st.subheader("❓ 청년정책 질문하기")
    
    # 채팅 기록 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # 채팅 기록 표시
    for message in st.session_state.chat_history:
        role = message.get("role")
        
        with st.chat_message(role):
            if role == "assistant":
                # 요약 표시
                if "summary" in message:
                    st.markdown(f'<div class="summary-box"><strong>📌 요약</strong><br>{message["summary"]}</div>', unsafe_allow_html=True)
                
                # 전체 답변 표시
                st.markdown(message["content"])
                
                # 검색된 정책 표시
                if "documents" in message and message["documents"]:
                    with st.expander(f"📊 검색된 정책 ({len(message['documents'])}개)", expanded=False):
                        for i, doc in enumerate(message["documents"][:5], 1):
                            metadata = doc.metadata
                            st.markdown(f"""
                            <div class="policy-card">
                                <strong>{i}. {metadata.get('정책명', 'N/A')}</strong><br>
                                📍 {metadata.get('지역', 'N/A')}<br>
                                🎯 {metadata.get('정책유형', 'N/A')}<br>
                                👥 연령: {metadata.get('연령', 'N/A')}<br>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(message["content"])
    
    # 질문 입력
    if question := st.chat_input("청년 정책에 대해 무엇이든 물어보세요!"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # RAG 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                if rag:
                    try:
                        result = rag.query(question)
                        
                        # 요약 표시
                        if "summary" in result:
                            st.markdown(f'<div class="summary-box"><strong>📌 요약</strong><br>{result["summary"]}</div>', unsafe_allow_html=True)
                        
                        # 전체 답변 표시
                        answer = result.get("answer", "답변을 생성할 수 없습니다.")
                        st.markdown(answer)
                        
                        # 검색된 정책 표시
                        documents = result.get("documents", [])
                        if documents:
                            with st.expander(f"📊 검색된 정책 ({len(documents)}개)", expanded=False):
                                for i, doc in enumerate(documents[:5], 1):
                                    metadata = doc.metadata
                                    st.markdown(f"""
                                    <div class="policy-card">
                                        <strong>{i}. {metadata.get('정책명', 'N/A')}</strong><br>
                                        📍 {metadata.get('지역', 'N/A')}<br>
                                        🎯 {metadata.get('정책유형', 'N/A')}<br>
                                        👥 연령: {metadata.get('연령', 'N/A')}<br>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # 채팅 기록에 추가
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "summary": result.get("summary", ""),
                            "documents": documents
                        })
                    except Exception as e:
                        error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = "❌ RAG 시스템이 초기화되지 않았습니다."
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# ========================================
# 메인 앱
# ========================================

def main():
    """메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="청년정책 Q&A 챗봇",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS 적용
    apply_custom_css()
    
    # 타이틀
    st.markdown('<h1 class="main-title">🎓 청년 정책 Q&A 챗봇</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 사이드바: RAG 파이프라인 로드
    with st.sidebar:
        st.header("⚙️ 시스템 설정")
        
        if st.button("🚀 RAG 파이프라인 로드", use_container_width=True, type="primary"):
            with st.spinner("RAG 파이프라인 초기화 중..."):
                rag = load_rag_pipeline()
                if rag:
                    st.session_state["rag_pipeline"] = rag
                    st.success("✅ RAG 파이프라인 로드 완료!")
                else:
                    st.error("❌ RAG 파이프라인 로드 실패")
        
        st.markdown("---")
        
        # 대화 초기화 버튼
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.chat_history = []
            if "rag_pipeline" in st.session_state:
                st.session_state.rag_pipeline.clear_memory()
            st.success("대화 기록이 초기화되었습니다!")
            st.rerun()
        
        st.markdown("---")
        
        # 파이프라인 정보
        if "rag_pipeline" in st.session_state:
            st.success("🟢 RAG 파이프라인 활성화")
            st.info("""
            **활성화된 기능:**
            - 🔍 MultiQuery (3개 쿼리 생성)
            - 📊 BM25 + Vector 검색 (40% + 60%)
            - 💬 대화 기록 (최근 3턴)
            - 📌 Chain of Thought 요약
            """)
        else:
            st.warning("⚠️ RAG 파이프라인이 로드되지 않았습니다.")
    
    # 메인: 질문 인터페이스
    if "rag_pipeline" in st.session_state:
        render_question_interface(st.session_state.rag_pipeline)
    else:
        st.info("👈 좌측 사이드바에서 **RAG 파이프라인 로드** 버튼을 클릭하여 시작하세요!")


if __name__ == "__main__":
    main()