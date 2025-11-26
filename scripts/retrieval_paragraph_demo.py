
# scripts/test_retrieval.py
# Test với chunks thật từ PDF, chunking by paragraph (split '\n\n') cho retrieval tốt hơn

import os
import PyPDF2
from backend.core.retrieval import index_documents, retrieve

# Hàm split text thành paragraphs (cải thiện chunking)
def split_into_paragraphs(text):
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:  # Merge short para
            current_chunk += " " + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.append(para)
    if current_chunk:
        chunks.append(current_chunk)
    return [p for p in chunks if p]  # Bỏ empty

# Hàm đọc PDF và tạo chunks (paragraph per chunk)
def create_chunks_from_pdf(file_path):
    chunks = []
    file_name = os.path.basename(file_path)
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            paragraphs = split_into_paragraphs(text)
            for para_id, para_text in enumerate(paragraphs, start=1):
                chunks.append({
                    'text': para_text,
                    'metadata': {'file_name': file_name, 'page': page_num, 'para_id': para_id}
                })
    return chunks

# Đường dẫn PDF
pdf1 = '../data/raw/Instructions for the safe use of medicines.pdf'
pdf2 = '../data/raw/medicalguidelines.pdf'

# Tạo chunks từ 2 PDF (bây giờ nhỏ hơn, chi tiết hơn)
chunks = create_chunks_from_pdf(pdf1) + create_chunks_from_pdf(pdf2)
print(f"Total chunks: {len(chunks)}")#(cải thiện từ per page sang per paragraph)

# Index chunks
index_documents(chunks)

# Test retrieve với query về thành phần thuốc/hướng dẫn (thay query bạn tự nghĩ)
query = "What is the composition of aspirin?"  # Ví dụ tra thành phần aspirin từ medicalguidelines
# query = "How to store medicines?"  # Ví dụ hướng dẫn lưu trữ từ Instructions
results = retrieve(query, top_k=3)
print("Top 3 chunks:")
for res in results:
    print(f"Text: {res['text'][:200]}... | Score: {res['score']} | Metadata: {res['metadata']}")