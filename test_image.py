import cv2
import numpy as np
from depth import DepthEngine

print("[INFO] Memuat ONNX Runtime Engine...")
# Matikan stream & record, aktifkan mode save
engine = DepthEngine(stream=False, save=False, record=False)

print("[INFO] Membaca gambar test.jpg...")
frame = cv2.imread("test.jpg")
frame = cv2.resize(frame, (engine._width, engine._height))

print("[INFO] Menjalankan AI DepthAnything...")
depth_map = engine.infer(frame)

print("[INFO] Menyimpan hasil...")
# Gabungkan gambar asli dan peta kedalaman bersebelahan
hasil = np.concatenate((frame, depth_map), axis=1)
cv2.imwrite("hasil_test.jpg", hasil)

print("[SUCCESS] Cek file hasil_test.jpg di folder Anda!")