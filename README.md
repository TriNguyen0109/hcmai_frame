# 🎥 Hệ Thống Baseline Video Indexing & Search (TransNetV2 + CLIP + BEiT-3 + FAISS)

Hệ thống lập chỉ mục và tìm kiếm video tự động theo ngữ nghĩa dựa trên quy trình 5 bước chuẩn bài báo khoa học. Hệ thống hỗ trợ đóng gói **GPU Docker** giúp bạn dễ dàng đưa lên server GPU để triển khai.

---

## 📌 1. Các Tính Năng & Quy Trình 5 Bước (Pipeline)

1. **Phát hiện chuyển cảnh (Scene Boundary Detection)**: Sử dụng mô hình **TransNetV2** để chia video thành các đoạn cảnh (*scenes*) chính xác theo mốc thời gian.
2. **Trích xuất khung hình chính (Keyframe Selection)**: Với mỗi cảnh, trích xuất chính xác 4 khung hình (*keyframes*) cách đều nhau.
3. **Trích xuất đặc trưng kép (Dual Feature Extraction)**: Trích xuất vector nhúng chuẩn hóa $L_2$ đồng thời từ 2 mô hình:
   - **CLIP** (`openai/clip-vit-base-patch32`): Lấy đặc trưng ngữ nghĩa tổng quát (*Coarse-grained*).
   - **BEiT-3 / BEiT** (`microsoft/beit-base-patch16-224`): Lấy đặc trưng chi tiết (*Fine-grained*).
4. **Loại bỏ trùng lặp (Near-duplicate Deduplication)**: Tính Cosine Similarity giữa các keyframe trong cùng 1 cảnh. Nếu độ tương đồng $> 0.9$, tự động loại bỏ frame trùng lặp để tối ưu dung lượng lưu trữ.
5. **Lưu trữ Vector & Metadata (Storage & Indexing)**:
   - Lưu trữ vector đặc trưng vào 2 chỉ mục **FAISS** riêng biệt (`index_clip`, `index_beit3`).
   - Lưu trữ thông tin chi tiết vào tệp **JSON Metadata**.

---

## 📂 2. Cấu Trúc Mã Nguồn

```
aicity/
├── TransNetV2/                  # Mã nguồn & mô hình phát hiện chuyển cảnh TransNetV2
├── src/                         # Các mô-đun chính của Pipeline
│   ├── scene_detector.py        # Bước 1: Phát hiện cảnh bằng TransNetV2
│   ├── keyframe_extractor.py    # Bước 2: Chọn 4 keyframe cho mỗi cảnh
│   ├── feature_extractor.py     # Bước 3: Trích xuất vector CLIP + BEiT-3
│   ├── deduplicator.py          # Bước 4: Lọc trùng lặp keyframe (> 0.9 Cosine Sim)
│   └── storage.py               # Bước 5: Tạo FAISS Index & lưu JSON Metadata
├── pipeline.py                  # CLI chính thực thi quá trình lập chỉ mục Video
├── search.py                    # CLI tìm kiếm ngữ nghĩa theo câu truy vấn (Query Search)
├── requirements.txt             # Danh sách thư viện Python phụ thuộc
├── Dockerfile                   # File build Docker chứa môi trường GPU CUDA 12.1 + FFmpeg
├── docker-compose.yml           # File khởi chạy Docker GPU bằng 1 lệnh duy nhất
└── README.md                    # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ 3. Hướng Dẫn Cài Đặt

### Cách 1: Triển khai bằng Docker GPU (Khuyên dùng cho Server)

Yêu cầu máy server đã cài **Docker** và **NVIDIA Container Toolkit**.

1. **Build Docker Image**:
   ```bash
   docker compose build
   # Hoặc: docker build -t video-indexing-baseline .
   ```

---

### Cách 2: Cài đặt trực tiếp trên máy cục bộ (Local Python Environment)

Yêu cầu **Python 3.10+** và công cụ hệ thống **FFmpeg** (đã thêm vào PATH).

1. **Cài đặt các thư viện Python**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Cài đặt gói TransNetV2**:
   ```bash
   pip install -e ./TransNetV2
   ```

---

## 🚀 4. Hướng Dẫn Sử Dụng

### Bước 1: Lập chỉ mục Video (Video Indexing)

Đưa một tệp video đầu vào (ví dụ: `K01_V001.mp4`) qua Pipeline 5 bước để trích xuất và lưu vector.

#### 🐳 Chạy bằng Docker trên Server:
```bash
docker run --gpus all \
  -v $(pwd):/app/data \
  -v $(pwd)/storage:/app/storage \
  video-indexing-baseline:latest \
  python pipeline.py --video /app/data/K01_V001.mp4 --output_dir /app/storage --device cuda
```

#### 💻 Chạy trực tiếp bằng Python:
```bash
python pipeline.py --video K01_V001.mp4 --output_dir ./storage --device cuda
```

*Các tham số tuỳ chọn*:
- `--video`: Đường dẫn tới file video đầu vào.
- `--output_dir`: Thư mục lưu kết quả metadata và FAISS index (Mặc định: `./storage`).
- `--device`: Thiết bị tính toán `cuda` (GPU) hoặc `cpu`.
- `--sim_threshold`: Ngưỡng lọc trùng lặp keyframe (Mặc định: `0.9`).

---

### Bước 2: Tìm kiếm theo Truy vấn (Semantic Search)

Nhập một câu truy vấn mô tả bằng văn bản để tìm kiếm các cảnh/keyframe phù hợp nhất trong Cơ sở dữ liệu Vector.

#### 🐳 Chạy bằng Docker trên Server:
```bash
docker run --gpus all \
  -v $(pwd)/storage:/app/storage \
  video-indexing-baseline:latest \
  python search.py --query "a red car driving on a highway" --storage_dir /app/storage --top_k 5
```

#### 💻 Chạy trực tiếp bằng Python:
```bash
python search.py --query "a red car driving on a highway" --storage_dir ./storage --top_k 5
```

---

## 📄 5. Định Dạng Dữ Liệu Đầu Ra (Output Format)

Sau khi chạy xong `pipeline.py`, trong thư mục `./storage/` sẽ xuất hiện các file:

1. **`index_clip.index`**: Chỉ mục FAISS lưu vector đặc trưng của CLIP.
2. **`index_beit3.index`**: Chỉ mục FAISS lưu vector đặc trưng của BEiT-3.
3. **`keyframes/{video_id}/`**: Thư mục chứa các ảnh keyframes dạng `.jpg`.
4. **`{video_id}_metadata.json`**: File JSON Metadata có định dạng chuẩn:

```json
{
  "video_id": "K01_V001",
  "scenes": [
    {
      "scene_id": "scene_K01_V001_000",
      "start_time": 0.0,
      "end_time": 4.5,
      "keyframes": [
        {
          "frame_id": "frame_K01_V001_s000_1",
          "timestamp": 1.1,
          "image_path": "./storage/keyframes/K01_V001/frame_K01_V001_s000_1.jpg",
          "clip_vector_id": 1001,
          "beit3_vector_id": 1001
        },
        {
          "frame_id": "frame_K01_V001_s000_3",
          "timestamp": 3.3,
          "image_path": "./storage/keyframes/K01_V001/frame_K01_V001_s000_3.jpg",
          "clip_vector_id": 1002,
          "beit3_vector_id": 1002
        }
      ]
    }
  ]
}
```
