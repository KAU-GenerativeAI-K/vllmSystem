import gradio as gr
import base64
import requests

# LLM 서버 URL
# LLM_SERVER_URL = "http://배포한주소.com"

# 파일 업로드 처리 기능 정의
def handle_file_upload(files):
    if not files:  # files가 없거나 비어있는 경우
        return []  # 빈 리스트 반환

    file_contents = []
    base64_previews = []  # Base64 데이터를 저장할 리스트
    for file in files:
        if file.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # 이미지 파일
            try:
                # 이미지 파일을 Base64로 변환
                with open(file.name, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                file_contents.append({"type": "image", "name": file.name, "content": encoded_image})
                base64_previews.append(f"File: {file.name}, Base64: {encoded_image[:50]}...")  # 50자 미리보기 (Base64 미리보기)
            except Exception as e:
                file_contents.append({"type": "error", "name": file.name, "content": str(e)})
        elif file.name.lower().endswith(".pdf"):  # PDF 파일
            try:
                # PDF 파일 그대로 저장
                file_contents.append({"type": "pdf", "name": file.name})
            except Exception as e:
                file_contents.append({"type": "error", "name": file.name, "content": str(e)})
        else:  # 지원하지 않는 파일 형식
            file_contents.append({"type": "unsupported", "name": file.name})
    return file_contents

# 챗봇 기능 정의
def chatbot(message, chat_history, file_contents):
    pdf_files = [file for file in file_contents if file['type'] == 'pdf'] # PDF 파일 여부 확인
    image_files = [file for file in file_contents if file['type'] == 'image'] # 이미지 파일 여부

    try:
        if pdf_files and image_files:  # PDF 파일 업로드, 이미지 파일 업르드 된 경우
            data = {
                "message": message,
                "pdf_files": pdf_files,
                "image_files": image_files,
                "chat_history": chat_history,
            }

            # response = requests.post(LLM_SERVER_URL, json=data) # 서버 요청
            # response_data = response.json()  # 서버 응답 데이터
            # response = responst_data["response"] # LLM 서버 응답
        elif pdf_files: # PDF 파일만 업로드된 경우
            data = {
                "message": message,
                "pdf_files": pdf_files,
                "chat_history": chat_history,
            }

            # response = requests.post(LLM_SERVER_URL, json=data) # 서버 요청
            # response_data = response.json()  # 서버 응답 데이터
            # response = responst_data["response"] # LLM 서버 응답
        elif image_files: # 이미지 파일만 업로드된 경우
            data = {
                "message": message,
                "image_files": image_files,
                "chat_history": chat_history,
            }

            # response = requests.post(LLM_SERVER_URL, json=data) # 서버 요청
            # response_data = response.json()  # 서버 응답 데이터
            # response = responst_data["response"] # LLM 서버 응답
        else: # 파일이 업로드되지 않은 경우
            data = {
                "message": message,
                "chat_history": chat_history,
            }
            # response = requests.post(LLM_SERVER_URL, json=data) # 서버 요청
            # response_data = response.json()  # 서버 응답 데이터
            # response = responst_data["response"] # LLM 서버 응답
    except Exception as e:
        response = f"LLM server 통신 에러: {e}"

    # 대화 기록에 사용자 메시지와 챗봇 응답 추가
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response}) 
    return "", chat_history

# Gradio 앱 생성
with gr.Blocks(title="팀K Q&A 시스템") as demo:
    # UI 제목 추가
    gr.Markdown("# 팀K Q&A 시스템")  # 웹사이트 상단에 제목 표시

    file_contents = gr.State([])  # 파일 정보를 상태로 저장 (파일 정보를 지속적으로 유지)
    
    # UI 레이아웃
    with gr.Row():
        with gr.Column(scale=3):  # 첫 번째 열: 챗봇 인터페이스
            chatbot_ui = gr.Chatbot(label="Chatbot", type="messages")
            user_message = gr.Textbox(label="Your Message", placeholder="Type a message...")
            send_button = gr.Button("Send")
        with gr.Column(scale=1):  # 두 번째 열: 파일 업로드
            upload_button = gr.File(label="Upload Files", file_types=None, file_count="multiple")
            file_output = gr.Textbox(label="Uploaded Files Info", interactive=False)

    # 파일 업로드 버튼 동작 정의
    upload_button.change(
        handle_file_upload,  # 파일 처리 함수 호출
        inputs=upload_button,  # 업로드된 파일을 입력으로 사용
        outputs=file_contents # 파일 정보를 상태로 저장
    )
    
    # 전송 버튼 동작 정의
    send_button.click(
        chatbot,  # 챗봇 함수 호출
        inputs=[user_message, chatbot_ui, file_contents],  # 사용자 입력, 대화 기록, 파일 정보
        outputs=[user_message, chatbot_ui]  # 메시지 초기화 및 대화 기록 갱신
    )

# 앱 실행
demo.launch()