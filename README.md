# 팀K Q&A 시스템

이 프로젝트는 PDF 문서와 이미지 파일을 기반으로 한 RAG(Retrieval-Augmented Generation) 기반 Q&A 시스템입니다. LLAVA 모델을 통해 이미지 기반 질문 응답을 지원하며, VLLM을 활용해 GPT 기반의 고속 처리와 효율적인 추론을 제공합니다. Colab 환경에서 LLAVA와 VLLM을 실행하여 클라우드 기반 서빙을 구현합니다.

---

## 주요 기능

- **PDF 및 이미지 기반 Q&A**:
  - PDF에서 텍스트를 추출하고 RAG 방식으로 질문에 답변합니다.
  - 이미지 파일을 업로드하면 LLAVA 모델을 통해 이미지를 분석하고 관련 질문에 답변합니다.

- **VLLM 기반 GPT 모델 서빙**:
  - [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 모델을 VLLM으로 서빙하여 빠르고 정확한 GPT 추론을 제공합니다.
  - Pyngrok을 사용해 Colab에서 실행된 모델을 외부에서 접근 가능하도록 설정합니다.

- **LLAVA 모델 통합**:
  - LLAVA (Large Language and Vision Assistant) 모델을 통해 이미지 기반 질문 응답을 지원합니다.

- **FAISS 검색**:
  - PDF에서 추출한 텍스트를 FAISS 벡터 데이터베이스에 저장하여 효율적인 검색을 제공합니다.

- **문서 청킹(Chunking)**:
  - PDF 텍스트를 단락 또는 문장 단위로 나누어 검색 및 추론의 정확성을 높입니다.

---

## 기술 스택

- **언어 및 프레임워크**:
  - Python
  - [Gradio](https://gradio.app) (UI 인터페이스)

- **텍스트 처리 및 임베딩 생성**:
  - PyPDF2 (PDF 텍스트 추출)
  - spaCy (문장 분할 및 NLP 처리)
  - SentenceTransformers (`sentence-transformers/all-MiniLM-L6-v2` 모델)

- **이미지 처리 및 LLM**:
  - LLAVA 모델 (이미지 기반 질문 응답)
  - VLLM (고속 LLM 추론)

- **검색 및 인덱싱**:
  - FAISS (Facebook AI Similarity Search)

---
