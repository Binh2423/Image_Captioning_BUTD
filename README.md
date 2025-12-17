# Image Captioning BUTD + UpDown — Hướng dẫn đầy đủ (WSL + Kaggle)

> Repo tham chiếu: **Binh2423/Image_Captioning_BUTD** (nhánh `main`)  
> Mục tiêu: Chạy demo caption ảnh trên WSL (Ubuntu) và **huấn luyện trên Kaggle** bằng **SCAN features** (không cần ảnh thô).

---

## 1) Tổng quan
- Trích đặc trưng **BUTD** (Bottom-Up Top-Down) bằng Detectron2 (C4 R-101, Visual Genome attr).  
- Sinh caption bằng decoder **UpDown 2-LSTM** (đã có hai checkpoint: `xe_best.pt` (CE) & `scst_best.pt` (SCST)).  
- Giao diện **Gradio** để so sánh CE vs SCST, chỉnh **beam search** (beam, length penalty, no repeat n-gram…).

---

## 2) Cấu trúc tối thiểu (local)
```
project/
├─ app_coco.py                  # App gradio (CE vs SCST, beam search UI)
├─ setup.sh                     # Script cài thư viện cho WSL
├─ requirements.txt
├─ README_WSL_FULL.md           # (tuỳ) Hướng dẫn WSL chi tiết
└─ checkpoints/                 # Đặt toàn bộ trọng số/từ vựng vào đây
   ├─ faster_rcnn_from_caffe_attr.pkl   # >200MB (BUTD R-101-C4 attr)
   ├─ faster_rcnn_R_101_C4_attr_caffemaxpool.yaml
   ├─ vocab_coco.json
   ├─ xe_best.pt
   └─ scst_best.pt
```

**Tải nhanh toàn bộ checkpoint (GDrive):**  
- Folder: `https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR?hl=vi`  
  → Tải về và **đặt hết vào `project/checkpoints/`**

**Link gốc file BUTD lớn (>200MB) nếu cần tải riêng:**  
- `http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl`

---

## 3) Cài đặt trên WSL (Ubuntu 22.04)
> Yêu cầu: đã có WSL + Ubuntu, GPU NVIDIA (tuỳ chọn).

```bash
# Clone hoặc copy source vào WSL
cd ~ && mkdir -p Image_Captioning_BUTD && cd Image_Captioning_BUTD

# (Nếu đã có source, bỏ qua bước này)
# git clone https://github.com/Binh2423/Image_Captioning_BUTD.git .

# Tạo & kích hoạt venv Python 3.10 (khuyến nghị)
sudo apt update && sudo apt install -y python3.10-venv python3-pip git
python3 -m venv .venv310
source .venv310/bin/activate

# Chạy script cài thư viện
bash setup.sh
```

**Đặt trọng số & vocab**:
- Tạo thư mục: `mkdir -p checkpoints`
- Chép các file từ Google Drive ở trên vào `checkpoints/`  
  Bắt buộc phải có:  
  `faster_rcnn_from_caffe_attr.pkl`, `faster_rcnn_R_101_C4_attr_caffemaxpool.yaml`, `vocab_coco.json`.  
  (Tuỳ chọn) CE/SCST: `xe_best.pt`, `scst_best.pt`.

---

## 4) Chạy demo Gradio (local)
```bash
source .venv310/bin/activate
python app_coco.py
# Mở trình duyệt: http://127.0.0.1:7860 (hoặc http://<IP-WSL>:7860)
```

**Biến môi trường (tuỳ chọn)** — nếu đặt file ở vị trí khác:
```bash
export BUTD_YAML="./checkpoints/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"
export BUTD_WEIGHT="./checkpoints/faster_rcnn_from_caffe_attr.pkl"
export BUTD_VOCAB="./checkpoints/vocab_coco.json"
export BUTD_CE_CKPT="./checkpoints/xe_best.pt"
export BUTD_SCST_CKPT="./checkpoints/scst_best.pt"
python app_coco.py
```

**Trong UI** có thể chỉnh:  
Top-K boxes vẽ, **beam size**, **length penalty**, **no-repeat n-gram**, **max length**…  
Mục “Timing” hiển thị thời gian **trích đặc trưng** và **giải mã**.

---

## 5) Huấn luyện trên Kaggle (dùng SCAN features)
> Không cần ảnh thô. Dữ liệu đặc trưng đã dựng sẵn: **SCAN features**

- Dataset: `https://www.kaggle.com/datasets/kuanghueilee/scan-features`  
- Notebook mẫu (đã kèm sẵn trong repo): `iamge-captioning-butd.ipynb` *(đúng tên file!)*

### Cách chạy nhanh
1. Truy cập Kaggle → **Code** → **New Notebook**.  
2. **Add data** → tìm `kuanghueilee/scan-features` và **Add** vào notebook.  
3. **GPU**: bật GPU T4/ P100 (Tuỳ quota).  
4. **Upload** file notebook `iamge-captioning-butd.ipynb` (nếu chưa có).  
5. **Run All** (thứ tự cell đã sắp). Notebook đã có hàm:
   - Tạo vocab, chia dữ liệu train/val/test theo COCO-Format (đối với captions),
   - Nạp SCAN features (thay cho ảnh thô),
   - Train **CE** trước, sau đó **SCST** (CIDEr reward) nếu bật phần SCST.
6. **Kết quả**:
   - Checkpoints được lưu vào `/kaggle/working/checkpoints/`:
     - `xe_best.pt` — huấn luyện Cross Entropy.
     - `scst_best.pt` — huấn luyện SCST.
   - File `*_val_pred.json` + log metric COCO (CIDEr, BLEU-4, METEOR, ROUGE_L…).

> Ghi chú:
> - SCST baseline dùng **beam search** (beam=5 mặc định), **reward = CIDEr** (đã có `safe_compute_cider` trong code).  
> - Bạn có thể chỉ chạy **CE** để lấy `xe_best.pt` nếu không đủ thời gian/ quota cho SCST.

### Dùng checkpoint huấn luyện từ Kaggle qua WSL
- Tải `xe_best.pt` và/hoặc `scst_best.pt` về máy → đặt vào `project/checkpoints/`.
- Chạy lại app Gradio (mục 4).

---

## 6) Lỗi phổ biến & mẹo
- **File >200MB không thể push GitHub** → không commit `faster_rcnn_from_caffe_attr.pkl`.  
  Chỉ để link tải:  
  - GDrive tổng: `https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR?hl=vi`  
  - UNC gốc: `http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl`
- **Detectron2** mismatch → script `setup.sh` đã ghim phiên bản tương thích Torch 2.0/2.1 + D2=0.6.  
- **Gradio báo localhost** → chạy `python app_coco.py` trong WSL rồi truy cập `http://127.0.0.1:7860`. Nếu vẫn chặn, bật `share=True` trong `launch()`.

---

## 7) License & trích dẫn
- Mã nguồn tham khảo: Detectron2 (Facebook Research), BUTD (airsplay/py-bottom-up-attention), COCO Eval.  
- Dữ liệu SCAN: Kaggle `kuanghueilee/scan-features`.  
- Chỉ dùng cho mục đích học tập/ nghiên cứu học thuật.

---

## 8) Liên hệ
- Nếu cần tối ưu tốc độ/ UI beam-search: bật **Cache features** trong UI và tăng/giảm `beam`, `no-repeat n-gram`.  
- Vấn đề cài đặt trên WSL: kiểm tra `setup.sh` và driver CUDA.
