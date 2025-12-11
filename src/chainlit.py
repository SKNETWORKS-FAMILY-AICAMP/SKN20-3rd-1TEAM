import chainlit as cl
import os
import sys
import random
import asyncio

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

    # 초기 로딩 메시지
    thinking_msg = await cl.Message(content="").send()

    # 진행 상태 표시를 위한 비동기 함수
    async def show_progress():
        # 총 예상 시간 (4~10초 사이 랜덤)
        total_time = random.uniform(5.0, 10.0)
        
        stages = [
            # (메시지, 최소비율, 최대비율)
            ("잠시만! 네 질문의 핵심이 뭔지 꼼꼼히 뜯어보는 중이야... 🤔", 0.3, 0.4), # 분석
            ("어떤 정책 데이터가 필요할지 작전을 짜고 있어! 🗺️", 0.6, 0.8),       # 계획
            ("방대한 정책 도서관에서 관련 서류를 싹~ 뒤지는 중! 📚", 0.4, 0.5), # 검색
            ("찾은 내용들을 바탕으로 꿀팁 답변을 작성하고 있어! ✍️", 0.2, 0.3)  # 생성
        ]

        try:
            current_elapsed = 0
            for i, (msg, min_ratio, max_ratio) in enumerate(stages):
                # 단계별 메시지 업데이트
                if i > 0:
                    # 이전 메시지는 그대로 두고 새 메시지로 업데이트 (또는 덮어쓰기)
                    # 여기서는 덮어쓰기 방식으로 구현
                    pass
                
                thinking_msg.content = msg
                await thinking_msg.update()
                
                # 머무를 시간 계산
                if i == len(stages) - 1:
                    # 마지막 단계는 남은 시간만큼 대기 (혹은 실제 완료까지 계속)
                    # 여기서는 일단 무한 대기 대신 넉넉히 대기하도록 설정하거나
                    # 루프가 끝나도 RAG가 안끝나면 마지막 메시지 유지
                    wait_time = total_time - current_elapsed
                    if wait_time < 1.0: wait_time = 1.0
                else:
                    ratio = random.uniform(min_ratio, max_ratio)
                    wait_time = total_time * ratio
                
                await asyncio.sleep(wait_time)
                current_elapsed += wait_time
                
        except asyncio.CancelledError:
            # 작업이 취소되면 루프 종료
            pass

    # 백그라운드 태스크로 진행 표시 시작
    progress_task = asyncio.create_task(show_progress())

    try:
        # RAG 쿼리 실행
        # Chainlit은 비동기이므로 make_async로 동기 함수 래핑
        result = await cl.make_async(rag.query)(user_query)
        
        # 실제 작업이 끝나면 진행 표시 태스크 취소
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

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
        progress_task.cancel()
        thinking_msg.content = f"오류가 발생했습니다: {str(e)}"
        await thinking_msg.update()
