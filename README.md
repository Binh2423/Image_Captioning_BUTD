# Image Captioning BUTD — Hướng dẫn chạy trên **WSL** (Ubuntu) với trọng số từ Google Drive

Repo: **Image_Captioning_BUTD** (branch `main`)  
Trọng số (weights) mình đã upload sẵn: **Google Drive folder** → https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR

> Mục tiêu: cài đủ môi trường trên **WSL Ubuntu**, tải **weights** về thư mục `checkpoints/`, chạy app Gradio `app_coco.py` so sánh CE vs SCST.

---

## 1) Chuẩn bị WSL
- **Windows 10/11** + **WSL2** + **Ubuntu 22.04** (khuyến nghị).
- (Tùy chọn) GPU NVIDIA + CUDA driver Windows. Nếu không có GPU, code vẫn chạy CPU (chậm hơn).

## 2) Lấy mã nguồn
```bash
# vào WSL
cd ~
git clone https://github.com/Binh2423/Image_Captioning_BUTD.git
cd Image_Captioning_BUTD
```

## 3) Cài thư viện (đã có `setup.sh`)
Script này sẽ:
- Tạo venv Python 3.10 tại `~/.venv310`.
- Cài `torch`/`torchvision` phù hợp, `detectron2==0.6`, `gradio==4.44.1`, và các deps cần thiết.
```bash
chmod +x setup.sh
./setup.sh
# kích hoạt venv (sau mỗi lần mở terminal mới)
source ~/.venv310/bin/activate
```

## 4) Tải trọng số & đặt đúng thư mục
Tạo thư mục và tải từ **Google Drive** (link ở đầu README):
```
checkpoints/
 ├─ faster_rcnn_from_caffe_attr.pkl          # BUTD detector (lớn >200MB)
 ├─ faster_rcnn_R_101_C4_attr_caffemaxpool.yaml  # file YAML cấu hình detector
 ├─ vocab_coco.json                           # vocab cho caption
 ├─ xe_best.pt                                # checkpoint caption (CE)
 └─ scst_best.pt                              # checkpoint caption (SCST)
# (Tuỳ chọn) objects_vocab.txt, attributes_vocab.txt nếu bạn muốn hiển thị nhãn/thuộc tính đúng của VG.
```

> Nếu chưa có YAML, bạn có thể dùng file YAML đã kèm trong repo (thường nằm ở `checkpoints/`) hoặc tự tải từ repo/nguồn bạn đã dùng huấn luyện.

## 5) Cấu hình biến môi trường (đường dẫn)
> Thay đổi đường dẫn theo máy bạn (*giữ nguyên nếu bạn dùng venv & cấu trúc phía trên*).
```bash
# bên trong WSL + venv đã kích hoạt
export BUTD_YAML="$PWD/checkpoints/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml"
export BUTD_WEIGHT="$PWD/checkpoints/faster_rcnn_from_caffe_attr.pkl"
export BUTD_VOCAB="$PWD/checkpoints/vocab_coco.json"
export BUTD_CE_CKPT="$PWD/checkpoints/xe_best.pt"
export BUTD_SCST_CKPT="$PWD/checkpoints/scst_best.pt"

# (tuỳ chọn) nếu có 2 file này:
# export BUTD_OBJ_VOCAB="$PWD/checkpoints/objects_vocab.txt"
# export BUTD_ATTR_VOCAB="$PWD/checkpoints/attributes_vocab.txt"

# để không phải set lại mỗi lần mở terminal, thêm vào ~/.bashrc:
echo 'export BUTD_YAML="'$BUTD_YAML'"'        >> ~/.bashrc
echo 'export BUTD_WEIGHT="'$BUTD_WEIGHT'"'    >> ~/.bashrc
echo 'export BUTD_VOCAB="'$BUTD_VOCAB'"'      >> ~/.bashrc
echo 'export BUTD_CE_CKPT="'$BUTD_CE_CKPT'"'  >> ~/.bashrc
echo 'export BUTD_SCST_CKPT="'$BUTD_SCST_CKPT'"' >> ~/.bashrc
```

## 6) Chạy ứng dụng
```bash
# đang ở thư mục repo + venv đang bật
python app_coco.py
# Mặc định: http://0.0.0.0:7860 (mở trình duyệt Windows: http://localhost:7860)
# Nếu môi trường chặn localhost, sửa trong app_coco.py: launch(share=True)
```

### Lưu ý UI
- Kéo **beam size / length penalty / no-repeat n-gram / max length** để thay đổi decode và caption **sẽ cập nhật** (đã bắt các sự kiện thay đổi).
- Slider **Top-K boxes to draw** chỉ ảnh hưởng hiển thị; số box trích đặc trưng luôn là `NUM_OBJECTS` (mặc định 36) bên dưới.

## 7) Troubleshooting ngắn
- **Không thấy GPU**: kiểm tra `nvidia-smi` bên Windows, WSL có “CUDA on WSL” (WSL2 + driver mới).
- **Lỗi Gradio “bool is not iterable”**: đảm bảo `gradio==4.44.1` (script đã pin) và `show_api=False` trong `launch()` (đã đặt sẵn).
- **Box chồng chéo quá nhiều**: code đã khử trùng lặp IoU; có thể hạ `SCORE_THRESH_TEST` hoặc tăng `NMS_THRESH_TEST` trong `build_butd_predictor()`.

---

## Ghi chú về trọng số lớn
- File **BUTD detector** (>200MB) không push lên GitHub. Hãy **tải từ Drive** rồi đặt vào `checkpoints/` như mục (4).
- Link trực tiếp (nếu bạn cần thêm nguồn):  
  - UNC NLP: http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl  
  - Google Drive (đầy đủ cả CE/SCST & vocab): https://drive.google.com/drive/folders/13q0RGBR-XyaHXQwd2LH7zw_7BmUC4MkR

Chúc bạn chạy thuận lợi!