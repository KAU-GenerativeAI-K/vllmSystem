import faiss
import pickle
from sentence_transformers import SentenceTransformer
import PyPDF2
import spacy

# PDF 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
    return text

# 텍스트 청킹
def chunk_text(text, method="paragraph", chunk_size=3):
    if method == "paragraph":
        chunks = [para.strip() for para in text.split("\n") if para.strip()]
    elif method == "sentence":
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    else:
        raise ValueError("Invalid method. Choose 'paragraph' or 'sentence'.")
    return chunks

# 임베딩 생성
def generate_embeddings(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings, model

# FAISS 인덱싱 및 저장
def create_faiss_index(embeddings, index_path="faiss_index.bin"):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

# 청크 저장
def save_chunks(chunks, chunk_path="chunks.pkl"):
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

# PDF 처리 및 인덱스 생성 함수
def process_pdf_and_create_index(pdf_file, index_path="faiss_index.bin", chunk_path="chunks.pkl"):
    text = extract_text_from_pdf(pdf_file.name)
    chunks = chunk_text(text, method="paragraph", chunk_size=3)
    embeddings, _ = generate_embeddings(chunks)
    create_faiss_index(embeddings, index_path)
    save_chunks(chunks, chunk_path)
    return "PDF 파일 처리 및 인덱스 생성 완료!"