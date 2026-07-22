import torch
import torch.nn as nn


class InitialCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(InitialCNN, self).__init__()

        # 1. Khối trích xuất đặc trưng (Feature Extraction) với 1 lớp Conv2d
        self.features = nn.Sequential(
            # Giả sử ảnh đầu vào có 3 kênh màu (RGB), kích thước 64x64
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Kích thước ảnh sẽ giảm đi một nửa: 64x64 -> 32x32
        )

        # 2. Khối phân loại (Classifier)
        # Sau MaxPool, kích thước ảnh là [16, 32, 32] -> Số chiều sau khi flatten = 16 * 32 * 32 = 16384
        self.classifier = nn.Sequential(
            nn.Linear(16 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)  # Đầu ra tương ứng với số lượng lớp cần phân loại
        )

    def forward(self, x):
        # In shape đầu vào để theo dõi
        print(f"Shape đầu vào: {x.shape}")

        # Đi qua khối Conv
        x = self.features(x)
        print(f"Shape sau Conv2d + MaxPool2d: {x.shape}")

        # Chuyển đổi tensor 4D thành vector 2D (giữ lại batch_size ở chiều 0)
        x = torch.flatten(x, start_dim=1)
        print(f"Shape sau torch.flatten: {x.shape}")

        # Đi qua khối Linear (Fully Connected)
        x = self.classifier(x)
        print(f"Shape đầu ra cuối cùng: {x.shape}")

        return x


# --- Kiểm tra hoạt động của Model ---
if __name__ == "__main__":
    # Khởi tạo mô hình (giả lập bài toán phân loại 10 lớp)
    model = InitialCNN(num_classes=10)

    # Tạo một batch dữ liệu giả lập: 1 ảnh, 3 kênh màu, kích thước 64x64
    sample_input = torch.randn(1, 3, 64, 64)

    # Chạy thử mô hình
    output = model(sample_input)