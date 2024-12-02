import gradio as gr
import base64
from openai import OpenAI
from pdf_processing import process_pdfs_and_create_index
from faiss_search import load_faiss_index, load_chunks, search_top_k_with_context
from sentence_transformers import SentenceTransformer
import os
import shutil

# OpenAI Client 설정
client = OpenAI(
    base_url="https://5f8e-104-198-224-49.ngrok-free.app/v1",
    api_key="token-abc123",
)

# RAG 시스템 설정
INDEX_PATH = "vectorDB/faiss_index.bin"
CHUNK_PATH = "vectorDB/chunks.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIR = "vectorDB"

# vectorDB 디렉토리 초기화 함수
def clear_vector_db():
    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
        print("vectorDB 디렉토리가 삭제되었습니다.")

# 파일 업로드 처리
def handle_file_upload(files):
    if not files:
        return [], []  # 항상 2 개의 값을 반환

    file_contents = []
    uploaded_images = []  # 업로드된 이미지를 표시하기 위한 리스트
    pdf_files = []  # PDF 파일만 모아 처리할 리스트
    
    # 디렉토리 생성 확인
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    for file in files:
        if file.name.lower().endswith(".pdf"):
            pdf_files.append(file)
            file_contents.append({"type": "pdf", "name": file.name})
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            uploaded_images.append(file)
            with open(file.name, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            file_contents.append({"type": "image", "name": file.name, "content": encoded_image})
        else:
            file_contents.append({"type": "unsupported", "name": file.name})

    # PDF 파일 처리 및 인덱스 생성
    if pdf_files:
        process_pdfs_and_create_index(pdf_files, INDEX_PATH, CHUNK_PATH)

    return file_contents, uploaded_images

# 파일 삭제 처리
def handle_file_delete(file_names, file_contents):
    if not file_names:
        file_names = []
    if not file_contents:
        file_contents = []
        
    updated_file_contents = [
        file for file in file_contents if file['name'] not in file_names
    ]
    updated_uploaded_images = [
        file['name'] for file in updated_file_contents if file['type'] == 'image'
    ]
    return updated_file_contents, updated_uploaded_images

# 채팅 히스토리 및 파일 초기화 처리
def reset_chat(file_contents):
    file_contents = []  # 파일 초기화
    chat_history = []   # 채팅 히스토리 초기화
    uploaded_files = None  # 업로드된 파일 상태 초기화
    return chat_history, file_contents, [], uploaded_files

# 챗봇 기능 정의
def chatbot(message, chat_history, file_contents):
    pdf_files = [file for file in file_contents if file['type'] == 'pdf']
    image_files = [file for file in file_contents if file['type'] == 'image']

    try:
        if not pdf_files:  # PDF 파일이 없는 경우
            if not image_files:  # 이미지 파일도 없는 경우
                # 일반 질문에 대한 기본 답변
                prompt = "you are a helpful assistant. 주의사항 : 한국어로만 답변할것."
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ]
            else:  # 이미지 파일만 있는 경우
                # 이미지 파일을 설명으로 추가
                image_descriptions = "\n".join([f"이미지 파일: {img['name']}" for img in image_files])
                prompt = (
                    "you are a helpful assistant. 주의사항 : 한국어로만 답변할것. "
                    "다음 이미지를 참고하여 질문에 답변하세요:\n" + image_descriptions
                )
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message}
                ]
        else:  # PDF 파일이 있는 경우
            # FAISS 인덱스 로드 및 쿼리 실행
            index = load_faiss_index(INDEX_PATH)
            chunks = load_chunks(CHUNK_PATH)
            model = SentenceTransformer(MODEL_NAME)
            search_results = search_top_k_with_context(index, message, model, chunks, k=5, context_range=3)
            
            # 레퍼런스 기반 프롬프트 생성
            context = "\n".join([result[0] for result in search_results])
            image_descriptions = (
                "\n".join([f"이미지 파일: {img['name']}" for img in image_files])
                if image_files else ""
            )

            prompt = (
                "사용자가 질문한 내용에 대해 답변을 생성해줘. 답변을 생성할 때는 다음의 조건을 반드시 따라야 해:\n"
                "1. 질문과 가장 관련 있는 문서의 내용만 사용해.\n"
                "2. 질문과 관련 없는 문서의 내용은 절대 사용하지 마.\n"
                "3. 주어진 문서들에 없는 내용은 절대 생성하지 마.\n"
                "4. 답변은 반드시 한국어로 작성해야 해.\n\n"
                "문서내용: {context}"
                "이미지파일: {image_descriptions}"
            )
            
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message}
            ]

        # OpenAI API 호출
        response = client.chat.completions.create(
            messages=messages,
            model="mistralai/Mistral-7B-Instruct-v0.2"
        )
        bot_response = response.choices[0].message.content

    except Exception as e:
        bot_response = f"LLM 서버 통신 에러: {e}"

    # 대화 기록에 사용자 메시지와 챗봇 응답 추가
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_response})
    return "", chat_history

# Gradio 앱 생성
with gr.Blocks(title="팀K Q&A 시스템") as demo:
    gr.Markdown("# 팀K Q&A 시스템")

    file_contents = gr.State([])
    chat_history = gr.State([])
    session_state = gr.State(value=None, delete_callback=clear_vector_db)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(label="Chatbot", type="messages")
            user_message = gr.Textbox(label="Your Message", placeholder="Type a message...")
            send_button = gr.Button("Send")
        with gr.Column(scale=1):
            upload_button = gr.File(label="Upload Files", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"], file_count="multiple")
            uploaded_images_ui = gr.Gallery(label="Uploaded Images", columns=1)  # 업로드된 이미지를 표시할 갤러리 추가
            reset_button = gr.Button("Reset Chat") 

    # 파일 업로드 이벤트
    upload_button.change(
        handle_file_upload,
        inputs=upload_button,
        outputs=[file_contents, uploaded_images_ui]  # 이미지 UI와 파일 내용을 출력
    )

    # 파일 삭제 이벤트
    upload_button.clear(
        handle_file_delete,
        inputs=[upload_button, file_contents],
        outputs=[file_contents, uploaded_images_ui]
    )
    
    # 메세지 전송 이벤트
    send_button.click(
        chatbot,
        inputs=[user_message, chatbot_ui, file_contents],
        outputs=[user_message, chatbot_ui]
    )

    # 초기화 버튼 이벤트
    reset_button.click(
        reset_chat,
        inputs=[file_contents],
        outputs=[chatbot_ui, file_contents, uploaded_images_ui, upload_button]
    )

demo.launch()