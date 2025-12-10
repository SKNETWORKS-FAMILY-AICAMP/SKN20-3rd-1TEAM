import chainlit as cl
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 임포트 해결
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.advanced_rag import initialize_rag_pipeline

@cl.on_chat_start
async def on_chat_start():
    """세션 시작 시 RAG 파이프라인을 초기화합니다."""
    try:
        # 데이터 경로 자동 계산
        vectordb_path = os.path.join(project_root, "data", "vectordb")
        
        # RAG 파이프라인 초기화
        rag = initialize_rag_pipeline(vectordb_path=vectordb_path)
        
    except Exception as e:
        await cl.Message(
            content=(
                "RAG 파이프라인 초기화에 실패했습니다.\n"
                f"- 오류: {e}\n"
                f"- 경로: {vectordb_path}"
            )
        ).send()
        return

    cl.user_session.set("rag", rag)

    await cl.Message(
        content=(
            "안녕! 나는 청년들의 든든한 정책 선배, 청년이음 선배봇🌟이야.\n"
            "주거, 월세, 일자리, 복지 정책 등 궁금한 점이 있으면 언제든지 나에게 물어봐!😺"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """사용자 메시지를 받아 RAG에 질의하고 답변을 반환합니다."""
    rag = cl.user_session.get("rag")

    if rag is None:
        await cl.Message(
            content="세션에 RAG 인스턴스가 없습니다. 새 채팅을 시작해 주세요."
        ).send()
        return

    user_query = message.content.strip()
    if not user_query:
        await cl.Message(content="질문 내용을 입력해 주세요.").send()
        return

    # 로딩 메시지
    thinking_msg = await cl.Message(content="정책 문서를 열심히 찾아보고 있어요... 🧐").send()

    try:
        # RAG 쿼리 실행
        # Chainlit은 비동기이므로 make_async로 동기 함수 래핑
        result = await cl.make_async(rag.query)(user_query)
        
        answer = result.get("answer", "죄송해요, 답변을 생성하지 못했어요. 😢")
        summary = result.get("summary", "")
        metadata = result.get("metadata", {})

        # 1. 답변 구성
        final_content = answer

        # 2. 요약이 있다면 추가
        if summary:
            final_content += f"\n\n---\n\n**[핵심 요약]**\n{summary}"

        # 3. CoT (생각의 과정) 시각화 - Chainlit Step 활용 (접이식 UI)
        if metadata:
            cot_content = ""
            
            # 다중 쿼리
            queries = metadata.get("queries", [])
            if queries:
                cot_content += "**1. 다중 쿼리 생성 (Multi-Query)**\n"
                for q in queries:
                    cot_content += f"- {q}\n"
                cot_content += "\n"

            # 지역 필터
            region_filter = metadata.get("region_filter")
            if region_filter:
                cot_content += "**2. 지역 필터링 (Region Filter)**\n"
                cot_content += f"```json\n{region_filter}\n```\n"
            
            # 검색 문서 수
            num_docs = metadata.get("num_docs_retrieved", 0)
            cot_content += f"**3. 검색된 문서 수**: {num_docs}개\n"
            
            # Step으로 출력 (접혀진 상태로 표시됨)
            async with cl.Step(name="🔍 선배봇의 생각 과정 보기") as step:
                step.output = cot_content

        # 응답 업데이트
        thinking_msg.content = final_content
        await thinking_msg.update()

    except Exception as e:
        thinking_msg.content = f"오류가 발생했습니다: {str(e)}"
        await thinking_msg.update()
