import gradio as gr
import base64
from openai import OpenAI
from pdf_processing import process_pdfs_and_create_index
from faiss_search import load_faiss_index, load_chunks, search_top_k_with_context
from sentence_transformers import SentenceTransformer
import os
import shutil
import time

# OpenAI Client 설정
client = OpenAI(
    base_url="https://6b6e-35-185-179-250.ngrok-free.app/v1",
    api_key="token-abc123",
)

models = client.models.list()
MODEL = models.data[0].id

# RAG 시스템 설정
INDEX_PATH = "vectorDB/faiss_index.bin"
CHUNK_PATH = "vectorDB/chunks.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_DIR = "vectorDB"


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
                extension = os.path.splitext(file.name)[1][1:].lower() # os.path.splitext로 확장자를 안전하게 추출
                image_url = f"data:image/{extension};base64,{encoded_image}"
            file_contents.append({"type": "image", "name": file.name, "url": image_url})
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
def chatbot(query, chat_history, file_contents):
    """챗봇에서 최신 파일 정보를 반영하여 메시지를 생성"""
    pdf_files = [file for file in file_contents if file['type'] == 'pdf']
    image_files = [file for file in file_contents if file['type'] == 'image']

    try:
        MAX_CONTEXT_LENGTH = 4
        truncated_history = chat_history[-MAX_CONTEXT_LENGTH:]

        if not pdf_files:
            if not image_files:
                messages = [{"role": "system", "content": "You are a helpful assistant."}]
                messages = [
                    {"role": item["role"], "content": item["content"]}
                    for item in truncated_history if "role" in item and "content" in item
                    ]
                messages.append({"role": "user", "content": query})

                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                response = chat_response.choices[0].message.content.strip()
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history
            else:
                if len(image_files) == 1:
                    # 이미지가 1개인 경우
                    image_url = image_files[0]['url']

                    # 히스토리를 메시지에 추가
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                    messages = [
                        {"role": item["role"], "content": item["content"]}
                        for item in truncated_history if "role" in item and "content" in item
                        ]
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    })

                    chat_response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                    )
                    response = chat_response.choices[0].message.content.strip()
                    chat_history.append({"role": "user", "content": query})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history
                else:
                    responses = []  # 여러 개의 이미지에 대한 응답을 저장할 리스트
                    # 이미지가 여러 개인 경우
                    for img in image_files:
                        image_url = img['url']
                        chat_response = client.chat.completions.create(
                            model=MODEL,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Describe the image in detail."},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": image_url,
                                        },
                                    },
                                ],
                            }],
                        )
                        responses.append(chat_response.choices[0].message.content.strip())

                    # 이미지 설명과 사용자 질문을 결합
                    combined_context = "Here are the detailed descriptions of the images:\n"
                    for i, response in enumerate(responses, start=1):
                        combined_context += f"Image {i}: {response}\n"

                    combined_context = f"User's question: {query}\n" + combined_context


                    # 히스토리를 메시지에 추가
                    messages = [{"role": "system", "content": "You are a multimodal assistant. Use the image descriptions to answer the user's question."}]
                    messages = [
                        {"role": item["role"], "content": item["content"]}
                        for item in truncated_history if "role" in item and "content" in item
                        ]
                    messages.append({"role": "user", "content": combined_context})

                    chat_response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                    )
                    response = chat_response.choices[0].message.content.strip()
                    chat_history.append({"role": "user", "content": query})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history
        else:
            index = load_faiss_index(INDEX_PATH)
            chunks = load_chunks(CHUNK_PATH)
            model = SentenceTransformer(MODEL_NAME)
            search_results = search_top_k_with_context(index, query, model, chunks, k=5, context_range=3)

            context = "\n\n".join([result[0] for result in search_results])
            if not image_files:
                prompt = """
                You are an AI that generates answers exclusively based on the provided references. Follow these instructions when crafting your response:

                1. Your answer must be based solely on the content of the provided references.
                2. Do not infer or generate any information that is not explicitly stated in the references.
                3. If the references do not contain relevant information to answer the question, respond with: "The references do not contain this information.
                """

                user_query =f"""
                Below are the references and the user's question:
                references:
                {context}

                Question:
                {query}
                """

                messages = [{"role": "system", "content": prompt}]
                messages = [
                    {"role": item["role"], "content": item["content"]}
                    for item in truncated_history if "role" in item and "content" in item
                    ]
                messages.append({"role": "user", "content": user_query})

                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                response = chat_response.choices[0].message.content.strip()
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history
            else:
                responses = []  # 여러 개의 이미지에 대한 응답을 저장할 리스트
                for img in image_files:
                    image_url = img['url']
                    chat_response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe the image in detail."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                    },
                                },
                            ],
                        }],
                    )
                    responses.append(chat_response.choices[0].message.content.strip())

                # 이미지 설명과 사용자 질문을 결합
                combined_context = "Here are the detailed descriptions of the images:\n"
                for i, response in enumerate(responses, start=1):
                    combined_context += f"Image {i}: {response}\n"

                prompt = """
                You are an AI that generates answers exclusively based on the provided references. Follow these instructions when crafting your response:

                1. Your answer must be based solely on the content of the provided references.
                2. Do not infer or generate any information that is not explicitly stated in the references.
                3. If the references do not contain relevant information to answer the question, respond with: "The references do not contain this information.
                """

                user_query =f"""
                Below are the references and the user's question:
                references:
                {context}

                image_references:
                {combined_context}

                Question:
                {query}
                """

                messages = [{"role": "system", "content": prompt}]
                messages = [
                    {"role": item["role"], "content": item["content"]}
                    for item in truncated_history if "role" in item and "content" in item
                    ]
                messages.append({"role": "user", "content": user_query})

                chat_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                )
                response = chat_response.choices[0].message.content.strip()
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": response})
                return "", chat_history

    except Exception as e:
        return f"LLM 서버 통신 에러: {e}"

# Gradio 앱 생성
with gr.Blocks(title="팀K Q&A 시스템") as demo:
    gr.Markdown("# 팀K Q&A 시스템")

    file_contents = gr.State([])
    chat_history = gr.State([])

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