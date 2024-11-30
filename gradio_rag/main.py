import gradio as gr
import base64
from openai import OpenAI
from pdf_processing import process_pdf_and_create_index
from faiss_search import load_faiss_index, load_chunks, search_top_k_with_context
from sentence_transformers import SentenceTransformer

# OpenAI Client 설정
client = OpenAI(
    base_url="https://5f8e-104-198-224-49.ngrok-free.app/v1",
    api_key="token-abc123",
)

# RAG 시스템 설정
INDEX_PATH = "vectorDB/faiss_index.bin"
CHUNK_PATH = "vectorDB/chunks.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 파일 업로드 처리
def handle_file_upload(files):
    if not files:
        return []

    file_contents = []
    for file in files:
        if file.name.lower().endswith(".pdf"):
            # PDF 처리 및 인덱스 생성
            process_pdf_and_create_index(file, INDEX_PATH, CHUNK_PATH)
            file_contents.append({"type": "pdf", "name": file.name})
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            with open(file.name, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            file_contents.append({"type": "image", "name": file.name, "content": encoded_image})
        else:
            file_contents.append({"type": "unsupported", "name": file.name})
    return file_contents

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
            search_results = search_top_k_with_context(index, message, model, chunks, k=5, context_range=1)
            
            # 레퍼런스 기반 프롬프트 생성
            context = "\n".join([result[0] for result in search_results])
            image_descriptions = (
                "\n".join([f"이미지 파일: {img['name']}" for img in image_files])
                if image_files else ""
            )
            prompt = (
                "you are a helpful assistant. 레퍼런스를 기반으로 답변할것. "
                "주의사항 : 한국어로만 답변할것.\n\nReferences:\n" + context + "\n" + image_descriptions
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

    with gr.Row():
        with gr.Column(scale=3):
            chatbot_ui = gr.Chatbot(label="Chatbot", type="messages")
            user_message = gr.Textbox(label="Your Message", placeholder="Type a message...")
            send_button = gr.Button("Send")
        with gr.Column(scale=1):
            upload_button = gr.File(label="Upload Files", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".gif"], file_count="multiple")
            file_output = gr.Textbox(label="Uploaded Files Info", interactive=False)

    upload_button.change(
        handle_file_upload,
        inputs=upload_button,
        outputs=file_contents
    )
    
    send_button.click(
        chatbot,
        inputs=[user_message, chatbot_ui, file_contents],
        outputs=[user_message, chatbot_ui]
    )

demo.launch()
