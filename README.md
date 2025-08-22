## XuNet Steganalysis – Quick Start

Ngắn gọn các bước cài đặt & chạy lại mô hình trên Windows (PowerShell / CMD).

### 1. Clone / Lấy mã nguồn
```cmd
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```

### 2. Tạo virtual environment (Python 3.10+)
```cmd
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3. Cài thư viện
CPU:
```cmd
pip install torch torchvision numpy pillow imageio
```
GPU (ví dụ CUDA 12.x):
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow imageio
```

### 4. Chuẩn bị dữ liệu
Tạo cấu trúc (đặt ảnh cover & stego tương ứng – tên không cần liên tục, hỗ trợ bmp/pgm/png/jpg):
```cmd
mkdir data
mkdir data\train_cover data\train_stego data\val_cover data\val_stego
```
Copy file ảnh vào 4 thư mục trên.

### 5. Chạy huấn luyện (ví dụ thông số)
```cmd
python train.py ^
 --cover_path data/train_cover ^
 --stego_path data/train_stego ^
 --valid_cover_path data/val_cover ^
 --valid_stego_path data/val_stego ^
 --train_size 1000 ^
 --val_size 200 ^
 --batch_size 16 ^
 --lr 0.001 ^
 --num_epochs 50 ^
 --checkpoints_dir checkpoints/
```

Script sẽ tự tạo thư mục `checkpoints/` và resume nếu đã có checkpoint (net_*.pt).

### 6. Đổi tên checkpoint cuối (tùy chọn)
```cmd
copy checkpoints\net_50.pt checkpoints\XuNet_model_weights.pt
```

### 7. Kiểm tra nhanh mô hình
```cmd
python - <<EOF
import torch; from model.model import XuNet
print(XuNet()(torch.randn(2,1,256,256)).shape)
EOF
```

### 8. Inference 1 ảnh đơn
```cmd
python - <<EOF
import torch, imageio.v2 as io
from model.model import XuNet
m = XuNet(); ckpt = torch.load('checkpoints/XuNet_model_weights.pt', weights_only=False, map_location='cpu')
m.load_state_dict(ckpt['model_state_dict']); m.eval()
img = io.imread('sample.pgm')
import numpy as np
if img.ndim==3: img = img[...,0]
tensor = torch.tensor(img).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
  out = m(tensor); prob = out.exp(); print('Prob (cover, stego):', prob.tolist())
EOF
```

### 9. Xem log huấn luyện
```cmd
type training.log
```

### 10. .gitignore tối thiểu (nếu cần tạo)
```cmd
echo data/>.gitignore
echo checkpoints/>>.gitignore
echo *.pt>>.gitignore
echo raw_cover/>>.gitignore
echo BOWS-2/>>.gitignore
```

### 11. Commit & Push
```cmd
git init
git add .
git commit -m "Init XuNet"
git remote add origin https://github.com/<your-org>/<your-repo>.git
git push -u origin main
```

### 12. Thoát môi trường
```cmd
deactivate
```

---
Tài liệu gốc bài báo: *Structural Design of Convolutional Neural Networks for Steganalysis (XuNet)*.

Ảnh kiến trúc (tùy chọn đặt file `xunet.png` cùng thư mục) nếu muốn hiển thị.
