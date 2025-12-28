# Phân tích dự án Video Subtitle Remover (VSR)

## 📋 Tổng quan dự án

**Video Subtitle Remover (VSR)** là một ứng dụng AI để xóa phụ đề cứng (hardcoded subtitles) khỏi video và ảnh. Dự án sử dụng các thuật toán deep learning để phát hiện và xóa phụ đề, sau đó điền vào vùng đã xóa bằng AI inpainting.

## 🏗️ Cấu trúc dự án

```
video-subtitle-remover/
├── backend/                    # Backend chính
│   ├── main.py                # Entry point CLI, chứa SubtitleRemover class
│   ├── config.py              # Cấu hình toàn cục (algorithms, parameters)
│   ├── inpaint/               # Các thuật toán inpainting
│   │   ├── sttn_inpaint.py    # STTN algorithm
│   │   ├── lama_inpaint.py    # LAMA algorithm
│   │   └── video_inpaint.py   # ProPainter algorithm
│   ├── models/                # Các mô hình AI
│   │   ├── big-lama/          # LAMA model
│   │   ├── sttn/              # STTN model
│   │   ├── video/             # ProPainter models
│   │   └── V4/ch_det/         # Text detection model (PaddleOCR)
│   ├── scenedetect/           # Scene detection utilities
│   └── tools/                 # Utility functions
├── gui.py                     # GUI interface (PySimpleGUI)
├── requirements.txt           # Python dependencies
└── test/                      # Test videos
```

## 🔑 Các thành phần chính

### 1. **SubtitleDetect** (`backend/main.py`)
- Phát hiện phụ đề trong video/ảnh
- Sử dụng PaddleOCR để phát hiện text
- Hỗ trợ chỉ định vùng phụ đề hoặc tự động phát hiện
- Xử lý scene detection để tối ưu xử lý

### 2. **SubtitleRemover** (`backend/main.py`)
- Class chính để xóa phụ đề
- Hỗ trợ 3 chế độ:
  - **STTN Mode**: Nhanh, tốt cho video người thật
  - **LAMA Mode**: Tốt cho ảnh và video hoạt hình
  - **ProPainter Mode**: Tốt nhất nhưng cần nhiều VRAM

### 3. **Các thuật toán Inpainting**

#### STTN (Spatial-Temporal Transformer Network)
- **Ưu điểm**: Nhanh, có thể bỏ qua detection
- **Nhược điểm**: Cần nhiều VRAM khi xử lý nhiều frame
- **Tham số quan trọng**:
  - `STTN_NEIGHBOR_STRIDE`: Bước nhảy giữa các frame tham chiếu
  - `STTN_REFERENCE_LENGTH`: Số lượng frame tham chiếu
  - `STTN_MAX_LOAD_NUM`: Số frame tối đa xử lý cùng lúc

#### LAMA (Large Mask Inpainting)
- **Ưu điểm**: Tốt cho ảnh tĩnh và video hoạt hình
- **Nhược điểm**: Không thể bỏ qua detection, chậm hơn STTN
- **Tham số**: `LAMA_SUPER_FAST`: Chế độ nhanh (giảm chất lượng)

#### ProPainter
- **Ưu điểm**: Chất lượng tốt nhất, xử lý tốt video chuyển động mạnh
- **Nhược điểm**: Rất chậm, cần nhiều VRAM (20GB+)
- **Tham số**: `PROPAINTER_MAX_LOAD_NUM`: Số frame tối đa

### 4. **Text Detection** (PaddleOCR)
- Phát hiện vùng text trong video
- Model: `V4/ch_det`
- Hỗ trợ ONNX runtime để tăng tốc

## ⚙️ Cấu hình (`backend/config.py`)

### Các tham số quan trọng:

```python
# Chọn thuật toán
MODE = InpaintMode.STTN  # hoặc LAMA, PROPAINTER

# STTN settings
STTN_SKIP_DETECTION = True  # Bỏ qua detection để tăng tốc
STTN_NEIGHBOR_STRIDE = 5
STTN_REFERENCE_LENGTH = 10
STTN_MAX_LOAD_NUM = 50

# LAMA settings
LAMA_SUPER_FAST = False

# ProPainter settings
PROPAINTER_MAX_LOAD_NUM = 70

# Text detection thresholds
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
PIXEL_TOLERANCE_X = 20
PIXEL_TOLERANCE_Y = 20
```

## 🚀 Cách sử dụng

### 1. CLI (Command Line)

```python
from backend.main import SubtitleRemover

# Tự động phát hiện phụ đề
remover = SubtitleRemover("video.mp4", sub_area=None)
remover.run()

# Chỉ định vùng phụ đề (ymin, ymax, xmin, xmax)
sub_area = (400, 500, 100, 900)  # Vùng ở dưới cùng
remover = SubtitleRemover("video.mp4", sub_area=sub_area)
remover.run()
```

### 2. GUI

```bash
python gui.py
```

### 3. Google Colab

Sử dụng file `Video_Subtitle_Remover_Colab.ipynb` đã được tạo.

## 📦 Dependencies

### Core libraries:
- **PyTorch**: Deep learning framework
- **PaddlePaddle**: Text detection (PaddleOCR)
- **OpenCV**: Video/image processing
- **FFmpeg**: Video encoding/decoding

### AI Models:
- **big-lama.pt**: LAMA inpainting model
- **infer_model.pth**: STTN model
- **ProPainter.pth**: ProPainter model
- **raft-things.pth**: RAFT optical flow
- **recurrent_flow_completion.pth**: Flow completion
- **inference.pdmodel/pdiparams**: Text detection model

## 🔧 Tối ưu hóa

### Tăng tốc độ:
1. Bật `STTN_SKIP_DETECTION = True`
2. Giảm `STTN_MAX_LOAD_NUM`
3. Sử dụng STTN thay vì ProPainter
4. Giảm độ phân giải video (nếu có thể)

### Cải thiện chất lượng:
1. Tăng `STTN_REFERENCE_LENGTH`
2. Tăng `STTN_NEIGHBOR_STRIDE`
3. Sử dụng ProPainter (nếu có đủ VRAM)
4. Tắt `STTN_SKIP_DETECTION` để phát hiện chính xác hơn

### Xử lý VRAM:
- Colab Free: ~15GB → Giảm `STTN_MAX_LOAD_NUM` xuống 30-40
- Colab Pro: ~25GB → Có thể dùng 50-70
- Local GPU: Tùy theo VRAM

## 📝 Workflow xử lý

1. **Load video**: Đọc video và lấy thông tin (fps, resolution, frame count)
2. **Text Detection** (nếu không skip):
   - Duyệt qua các frame
   - Sử dụng PaddleOCR để phát hiện text
   - Lưu vị trí phụ đề theo frame
3. **Scene Detection**: Phát hiện scene changes để tối ưu xử lý
4. **Inpainting**:
   - Chia video thành các đoạn
   - Xử lý từng đoạn với model inpainting
   - Điền vào vùng đã xóa phụ đề
5. **Merge audio**: Gộp audio gốc vào video đã xử lý
6. **Output**: Lưu video kết quả

## 🐛 Xử lý lỗi thường gặp

### 1. CUDA out of memory
- **Giải pháp**: Giảm `STTN_MAX_LOAD_NUM` hoặc `PROPAINTER_MAX_LOAD_NUM`
- Hoặc sử dụng LAMA mode

### 2. Models không tìm thấy
- **Giải pháp**: Tải models từ GitHub releases
- Hoặc sử dụng `filesplit` để merge các file split

### 3. FFmpeg không tìm thấy
- **Giải pháp**: Cài đặt FFmpeg hoặc chỉ định đường dẫn trong config

### 4. Import errors
- **Giải pháp**: Cài đặt lại dependencies từ `requirements.txt`

## 📊 So sánh các thuật toán

| Thuật toán | Tốc độ | Chất lượng | VRAM | Tốt cho |
|-----------|--------|-----------|------|---------|
| STTN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Trung bình | Video người thật |
| LAMA | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Thấp | Ảnh, video hoạt hình |
| ProPainter | ⭐⭐ | ⭐⭐⭐⭐⭐ | Cao | Video chuyển động mạnh |

## 🎯 Use Cases

1. **Xóa phụ đề từ video YouTube**: Tải video có phụ đề cứng, xóa và tạo lại
2. **Xóa watermark**: Xóa logo/watermark khỏi video
3. **Xử lý ảnh**: Xóa text khỏi ảnh
4. **Batch processing**: Xử lý nhiều video cùng lúc

## 📚 Tài liệu tham khảo

- GitHub: https://github.com/YaoFANGUK/video-subtitle-remover
- STTN Paper: Xem trong `design/paper_sttn.pdf`
- LAMA Paper: Xem trong `design/paper_intro.pdf`

## 🔄 Cập nhật

- **Version**: 1.1.1
- **Python**: 3.11+
- **PyTorch**: 2.7.0
- **PaddlePaddle**: 3.0.0

---

**Lưu ý**: Dự án này yêu cầu GPU để chạy hiệu quả. CPU có thể chạy nhưng rất chậm.

