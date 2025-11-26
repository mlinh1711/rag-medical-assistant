## 1. Embedding Model
### Các lựa chọn đã nghiên cứu:
- `text-embedding-ada-002` (OpenAI): chất lượng cao nhưng cần API key, tốn tiền.
- `all-mpnet-base-v2`: chất lượng tốt hơn MiniLM nhưng chậm hơn ~3x.
- `all-MiniLM-L6-v2` (Sentence-Transformers): **ĐỀ XUẤT CHỌN CÁI NÀY**

### Lý do chọn all-MiniLM-L6-v2:
- Kích thước nhỏ (80MB), chạy nhanh trên laptop thường.
- Độ chính xác đủ tốt cho tài liệu y khoa tiếng Anh.
- Miễn phí 100%, không cần API key.
- Hỗ trợ multilingual (nếu sau này có PDF tiếng Việt).
- Được cộng đồng RAG dùng rất nhiều (top 1 local embedding 2025).

Cài đặt: `pip install sentence-transformers`

## 2. Vector Store
### Các lựa chọn đã nghiên cứu:
- FAISS (Facebook): nhanh nhất nhưng phải tự quản lý file index.
- Pinecone: cloud, dễ scale nhưng cần API key + tốn tiền.
- Chroma: **ĐỀ XUẤT CHỌN CHROMA**

### Lý do chọn Chroma:
- Siêu dễ dùng với LangChain (chỉ 3 dòng code là lưu được).
- Tự động lưu xuống thư mục `./chroma_db` (persistent, tắt máy vẫn giữ).
- Local 100%, không cần internet sau khi tải model.
- Hỗ trợ metadata tốt (file_name, page_number) → frontend hiển thị nguồn đẹp.
- Cộng đồng lớn, debug dễ.

Cài đặt: `pip install chromadb`

## 3. Cách dùng trong code (gợi ý cho ngày 3)
```python
from sentence_transformers import SentenceTransformer
import chromadb

# Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("medical_docs")