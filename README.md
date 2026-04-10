# [WACV 2024] MotionAGFormer & 3D Fall Detection Pipeline 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) 
[![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2310.16288)

Dự án này là phiên bản mở rộng của **MotionAGFormer** (WACV 2024), được tích hợp thêm module **Phân loại hành động 3D (ST-GCN++)** chuyên dụng cho việc **Phát hiện hành động Ngã (Fall Detection)** từ video monocular.

---

## ✨ Tính năng nổi bật (Enhanced Features)

1.  **Lightning-fast 2D Pipeline**: Sử dụng **YOLOv8-pose** kết hợp **ByteTrack** để trích xuất và theo dõi khung xương 2D cực nhanh và ổn định.
2.  **Advanced 3D Reconstruction**: Kế thừa sức mạnh của MotionAGFormer để tái tạo tư thế 3D từ chuỗi 2D.
3.  **Real-world Root Recovery**: Thuật toán khôi phục tọa độ gốc 3D dựa trên độ sụt giảm hông (Hip descent) trong không gian 2D, cho phép quan sát quỹ đạo ngã thực tế.
4.  **Action Classification**: Tích hợp **ST-GCN++** huấn luyện trên bộ dữ liệu NTU-RGB+D để nhận diện chính xác các hành động: *Falling, Sitting, Standing, v.v.*
5.  **Professional Visualization**: Xuất video demo song song (Side-by-side) giữa video gốc và không gian 3D World-aligned.

---

## 🛠 Cài đặt môi trường (Setup)

### 1. Yêu cầu hệ thống
- **OS**: Windows/Linux
- **Python**: 3.8.10+
- **CUDA**: 11.8 / 12.1 (khuyên dùng để chạy GPU)

### 2. Cài đặt thư viện
```bash
# Cài đặt các thư viện cơ bản
pip install -r requirements.txt

# Cài đặt MMCV & PYSKL (Bắt buộc cho module Phân loại)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
# Sau đó cài đặt pyskl từ thư mục dự án
cd pyskl
pip install -e .
cd ..
```

### 3. Tải Pre-trained Models
Để hệ thống hoạt động, bạn cần đặt các file model vào đúng vị trí:
- **MotionAGFormer**: Tải từ [đây](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) và đặt vào thư mục `./checkpoint/`.
- **ST-GCN++ (Fall Detection)**: Đặt file `j.pth` vào thư mục `./pretrained/`.

---

## 🚀 Hướng dẫn vận hành (Quick Start)

Hệ thống cung cấp một pipeline tự động hoàn toàn từ Video đầu vào -> Video nhận diện hành động.

### Bước 1: Chuẩn bị video
Hãy đặt tệp video của bạn (định dạng `.mp4`, `.avi`) vào thư mục:  
`./demo/video/`

### Bước 2: Chạy lệnh Demo
Mở Terminal và chạy lệnh sau:
```powershell
python demo/vis.py --video your_video_name.mp4 --gpu 0
```
*Tham số:*
- `--video`: Tên file video trong thư mục `demo/video/`.
- `--gpu`: ID của card đồ họa (mặc định là 0).

### Bước 3: Xem kết quả
Sau khi chạy xong, kết quả sẽ nằm tại:
`./demo/output/your_video_name/your_video_name_demo.mp4`

Ngoài ra, bạn có thể quan sát kết quả phân loại thời gian thực trên Console log:
```text
>>>> ACTION DETECTED: FALLING (Score: 0.9845) <<<<
```

---

## 📂 Cơ cấu thư mục quan trọng

- `demo/vis.py`: Script chính điều phối toàn bộ pipeline (Detect 2D -> lifting 3D -> Classify).
- `classify_stgcnpp.py`: Module xử lý phân loại hành động sử dụng ST-GCN++.
- `pyskl/`: Thư viện lõi cho việc xử lý khung xương hành động.
- `checkpoint/`: Nơi chứa trọng số mô hình MotionAGFormer.
- `pretrained/`: Nơi chứa trọng số mô hình phân loại (ST-GCN++).

---

## 📝 Chú thích về Logic phân loại
Hệ thống sử dụng mô hình **ST-GCN++ (Joint stream)**. Để đạt kết quả tốt nhất:
- Subject nên xuất hiện rõ ràng trong khung hình.
- Khoảng cách từ camera đến subject nên ổn định (hỗ trợ tốt nhất cho góc nhìn Overhead/Góc nghiêng).
- Hành động ngã nên được thực hiện trong ít nhất 1-2 giây của clip.

---

## 🙏 Lời cảm ơn (Acknowledgement)
Dự án có sử dụng mã nguồn và ý tưởng từ:
- [MotionAGFormer Original Repository](https://github.com/hoanhv-iclever/MotionAGFormer)
- [PYSKL (OpenMMLab)](https://github.com/open-mmlab/pyskl)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)

---
*Chúc bạn có những trải nghiệm tuyệt vời với hệ thống Phát hiện Ngã 3D!* 🏁🚀