# Image Captioning (BUTD + UpDown) — Hướng dẫn đầy đủ (WSL & Kaggle, tiếng Việt)

> **Mục tiêu**: Chạy demo caption ảnh (so sánh CE vs SCST), và *tuỳ chọn* huấn luyện trên Kaggle dùng **SCAN features** (không cần ảnh thô). Tài liệu này gọn, thực dụng, dành cho **WSL (Ubuntu)** và **Kaggle**.

---

## 0) Cấu trúc dự án (tối thiểu)
```
.
├─ app_coco.py                 # App Gradio (đã patch cho Detectron2==0.6 và UI beam search)
├─ setup.sh                    # Script cài môi trường tự động cho WSL
├─ requirements.txt            # Thư viện nền
├─ README_FULL_VI.md           # (file này)
└─ checkpoints/                # Nơi để trọng số & vocab (tạo khi tải về)
   ├─ faster_rcnn_from_caffe_attr.pkl  # Trọng số detector BUTD (>=200MB)
   ├─ faster_rcnn_R_101_C4_attr_caffemaxpool.yaml
   ├─ vocab_coco.json
   ├─ xe_best.pt               # CE checkpoint (tùy chọn)
   └─ scst_best.pt             # SCST checkpoint (tùy chọn)
```

---

## 1) Chuẩn bị **checkpoints**

### Cách A — Dùng Google Drive (khuyên dùng)
Bạn đã có đầy đủ file tại thư mục Drive này (public của bạn):
- Drive: **https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR**

Tải các file về máy và đặt vào `./checkpoints/` với đúng tên:
- `faster_rcnn_from_caffe_attr.pkl`  (trên 200MB)
- `faster_rcnn_R_101_C4_attr_caffemaxpool.yaml`
- `vocab_coco.json`
- (tuỳ chọn) `xe_best.pt`, `scst_best.pt`

> **Lưu ý**: repo GitHub không nên commit file >200MB. Hãy để link tải trong README (như trên).

### Cách B — Tải trực tiếp BUTD detector
- PTH: **http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl**
- Sau khi tải, chép vào `./checkpoints/faster_rcnn_from_caffe_attr.pkl`

---

## 2) Cài đặt môi trường trên **WSL (Ubuntu)**

> Yêu cầu: WSL2 + Ubuntu 22.04; đã cài `git`, `curl`. Khuyến nghị driver/NVIDIA nếu dùng CUDA.

1. Cấp quyền & chạy script:
```bash
chmod +x ./setup.sh
./setup.sh
```

2. Script sẽ tự động:
- Cập nhật apt, cài `build-essential`, `cmake`, `git`, `ffmpeg`, `libgl1`, `libglib2.0-0`, v.v.
- Tạo **virtualenv** Python 3.10 tại `~/.venv310` (tên như trong app).
- Cài **torch**/**torchvision** matching CUDA (hoặc CPU), **detectron2==0.6** (build từ source) và deps phù hợp (`fvcore<0.1.6`, `iopath<0.1.10`, `pycocotools`, `opencv-python`, `gradio==4.44.1`, ...).
- Pin `numpy<2` để tránh lỗi binary.
- Cài *requirements.txt* của dự án.

3. Kích hoạt môi trường (mỗi lần mở shell mới):
```bash
source ~/.venv310/bin/activate
```

4. Kiểm tra:
```bash
python -c "import torch, detectron2, gradio; print(torch.__version__); print('detectron2 OK')"
```

---

## 3) Chạy **demo** Caption (CE vs SCST)

> Yêu cầu đã có file trong `./checkpoints/`. Đổi biến môi trường nếu cần (dưới đây là mặc định).

### 3.1 Biến môi trường (mặc định trong `app_coco.py`)
```bash
export BUTD_YAML=./checkpoints/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml
export BUTD_WEIGHT=./checkpoints/faster_rcnn_from_caffe_attr.pkl
export BUTD_VOCAB=./checkpoints/vocab_coco.json
export BUTD_CE_CKPT=./checkpoints/xe_best.pt        # nếu có
export BUTD_SCST_CKPT=./checkpoints/scst_best.pt    # nếu có
```

> Nếu bạn **chưa có** `xe_best.pt` / `scst_best.pt`, app vẫn chạy và hiển thị nhưng chỉ bằng decoder tải được. Hãy cung cấp cả 2 để so sánh CE vs SCST.

### 3.2 Chạy app
```bash
source ~/.venv310/bin/activate
python app_coco.py
```
- Mặc định listen ở `http://0.0.0.0:7860`
- Nếu localhost bị chặn, bật chế độ share:
  ```bash
  python app_coco.py --share  # (nếu bạn hỗ trợ flag; hoặc chỉnh trong code: share=True)
  ```

### 3.3 Lưu ý UI
- Thay đổi **Beam size / Length penalty / No-repeat / Max length** -> caption **được cập nhật** khi bạn:
  - Nhấn **Run**, hoặc
  - Bật chế độ **auto** (file đã bind `.change` trên các control — chỉ cần đã có ảnh).
- **Top-K boxes to draw** chỉ ảnh hưởng trực quan; extractor vẫn dùng K chuẩn (36).

---

## 4) Huấn luyện trên **Kaggle** với **SCAN features** (không dùng ảnh thô)

> Dùng dataset: **https://www.kaggle.com/datasets/kuanghueilee/scan-features**  
> Ưu điểm: giảm chi phí I/O, không cần Detectron2 trên Kaggle; chỉ train decoder CE/SCST.

### 4.1 Thiết lập Notebook (Kaggle)
- Tạo Notebook mới (GPU bật nếu muốn tốc độ nhanh hơn).
- Thêm **Dataset**: `kuanghueilee/scan-features` vào Notebook (button “Add data”).
- Upload notebook của bạn (nếu đã có). Nếu dùng notebook sẵn của repo, bảo đảm các cell sau có mặt:
  1. **Mount dataset**: trỏ tới `/kaggle/input/scan-features/`.
  2. **Tạo vocab / split**: (đã có sẵn hàm trong notebook). Sinh `vocab_coco.json`, annotations `train/val` theo COCO.
  3. **CE training**: Teacher forcing, lưu `xe_best.pt` vào `/kaggle/working/checkpoints/`.
  4. **SCST training**: Dùng CIDEr làm reward, baseline = beam search, lưu `scst_best.pt`.
  5. **Xuất file**: dùng `Output` của Kaggle để tải về hai checkpoint.

> **Quan trọng**: Các đoạn code train của bạn phải đọc **features** (~`*.npy`, `*.pt`) từ dataset SCAN và không load ảnh thô. Hãy đảm bảo `DataLoader` / `predict_from_loader` tương thích.

### 4.2 Tham số gợi ý (đủ minh hoạ)
- **CE**: `LR=1e-3`, `batch_size=128`, `epochs=10~20`, `label_smoothing=0.1`
- **SCST**: `LR=1e-5`, `epochs=5~30`, `SCST_MAXLEN=30`, `BEAM_BASELINE=5`, reward **CIDEr**

> SCST đã có helper `safe_compute_cider` trong mã. Khi lưu, hãy đặt đúng tên: `xe_best.pt` / `scst_best.pt`.

### 4.3 Dùng checkpoint train xong trên máy local/WSL
- Copy về `./checkpoints/xe_best.pt` và `./checkpoints/scst_best.pt` (local).
- Chạy lại app (mục 3) để so sánh CE vs SCST.

---

## 5) Mẹo hiệu năng & chất lượng
- **Extractor nhanh hơn**: bật `torch.backends.cudnn.benchmark = True` (đã bật) và AMP (`torch.amp.autocast`) trong detector.
- **Giảm box trùng**: đã thêm bước *khử trùng lặp IoU* và tăng score threshold/NMS (đỡ “một đối tượng nhiều box”).
- **Beam search**: `beam=3~5`, `len_penalty~1.0→1.2`, `no_repeat_ngram=1~3` cho caption mượt hơn.

---

## 6) Khắc phục lỗi thường gặp
- **Detectron2 skip weights / shape mismatch**: đúng, vì checkpoint Caffe-style; code đã patch tuyến suy luận tương thích.
- **Gradio “bool is not iterable / schema”**: đã patch vào `app_coco.py` (vô hiệu đường dẫn schema gây lỗi).
- **NumPy 2.x**: đã pin `numpy<2` trong cài đặt.
- **Localhost không vào được**: dùng `share=True` khi launch Gradio.

---

## 7) Gợi ý commit lên GitHub
- **Không** commit file >200MB (đặc biệt `.pkl` detector, các `.pt` lớn). Thay thế bằng **link tải** (Drive / URL UNC).
- Ví dụ trong README:
  - UNC: `http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl`
  - Drive: `https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR`

---

## 8) Chạy nhanh (tóm tắt)
```bash
# 1) Tải checkpoints vào ./checkpoints/
#    - UNC URL (detector): http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl
#    - Hoặc Drive đầy đủ:  https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR

# 2) Cài môi trường (WSL)
chmod +x setup.sh && ./setup.sh
source ~/.venv310/bin/activate

# 3) Chạy app
python app_coco.py
# Mở http://0.0.0.0:7860, upload ảnh → chỉnh beam/no-repeat/len_pen → xem CE vs SCST
```

---

## 9) Liên hệ
Nếu bạn vẫn gặp lỗi môi trường, hãy gửi:
- OS/WSL version, GPU/CUDA
- `python -V`, `pip list | grep -E "torch|torchvision|detectron2|gradio|numpy"`
- Log lỗi đầy đủ khi chạy `python app_coco.py`

Chúc bạn chạy mượt! 🚀
