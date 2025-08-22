import torch
from model.model import XuNet

model = XuNet()
ckpt = torch.load('./checkpoints/XuNet_model_weights.pt', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

print('✅ Training hoàn thành!')
print(f'📊 Kết quả cuối cùng:')
print(f'   - Epoch: {ckpt["epoch"]}')
print(f'   - Train Accuracy: {ckpt["train_accuracy"]:.1f}%')
print(f'   - Valid Accuracy: {ckpt["valid_accuracy"]:.1f}%')
print(f'   - Train Loss: {ckpt["train_loss"]:.5f}')
print(f'   - Valid Loss: {ckpt["valid_loss"]:.5f}')
print(f'💾 File weights: checkpoints/XuNet_model_weights.pt')
print('🎯 Model sẵn sàng để sử dụng!')
