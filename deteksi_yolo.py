"""Aplikasi deteksi objek dengan YOLOv8.

Cara pakai (di folder Pertemuan10, setelah menjalankan: venv):
  python tugas/deteksi_yolo.py gambar1.jpeg     -> deteksi pada gambar
  python tugas/deteksi_yolo.py video_sampel.mp4 -> deteksi pada video
  python tugas/deteksi_yolo.py                  -> deteksi live dari kamera (tekan q untuk berhenti)

Hasil disimpan di folder "tugas/hasil/".
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

FOLDER_TUGAS = Path(__file__).parent  # model & hasil selalu di folder tugas
FOLDER_HASIL = FOLDER_TUGAS / "hasil"
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# yolo11m jauh lebih akurat dari yolov8n (mAP COCO 51.5 vs 37.3), tapi lebih lambat.
# Kalau video/kamera terasa patah-patah, ganti ke "yolo11s.pt".
model = YOLO(FOLDER_TUGAS / "yolo11m.pt")
CONF = 0.4  # buang tebakan yang kurang yakin (default YOLO 0.25 -> banyak salah deteksi)


def ringkas(hasil):
    """Ubah hasil deteksi jadi teks, misal: '6 person, 2 potted plant'."""
    jumlah = Counter(hasil.names[int(c)] for c in hasil.boxes.cls)
    return ", ".join(f"{n} {nama}" for nama, n in jumlah.items()) or "tidak ada objek"


def deteksi_gambar(path, tampil=True):
    hasil = model(path, conf=CONF, verbose=False)[0]
    gambar = hasil.plot()  # gambar + kotak deteksi

    FOLDER_HASIL.mkdir(exist_ok=True)
    keluaran = FOLDER_HASIL / f"hasil_{Path(path).name}"
    cv2.imwrite(str(keluaran), gambar)
    print(f"Terdeteksi: {ringkas(hasil)}")
    print(f"Disimpan ke: {keluaran}")

    if tampil:
        cv2.imshow("Hasil Deteksi YOLO (tekan tombol apa saja untuk menutup)", gambar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return hasil


def deteksi_video(sumber, tampil=True):
    """sumber = path video, atau 0 untuk kamera."""
    cap = cv2.VideoCapture(sumber)
    if not cap.isOpened():
        sys.exit(f"Gagal membuka {'kamera' if sumber == 0 else sumber}. "
                 "Untuk kamera: izinkan akses kamera untuk Terminal/VS Code di System Settings.")

    FOLDER_HASIL.mkdir(exist_ok=True)
    nama = "kamera.mp4" if sumber == 0 else f"hasil_{Path(sumber).stem}.mp4"
    keluaran = FOLDER_HASIL / nama
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    penulis = cv2.VideoWriter(str(keluaran), cv2.VideoWriter_fourcc(*"mp4v"), fps, (lebar, tinggi))

    frame_ke = 0
    while True:
        ada, frame = cap.read()
        if not ada:
            break
        hasil = model(frame, conf=CONF, verbose=False)[0]
        gambar = hasil.plot()
        cv2.putText(gambar, ringkas(hasil), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        penulis.write(gambar)
        frame_ke += 1

        if tampil:
            cv2.imshow("Deteksi YOLO (tekan q untuk berhenti)", gambar)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    penulis.release()
    cv2.destroyAllWindows()
    print(f"Selesai: {frame_ke} frame diproses. Disimpan ke: {keluaran}")
    return frame_ke


if __name__ == "__main__":
    if len(sys.argv) < 2:
        deteksi_video(0)  # tanpa argumen -> kamera
    else:
        sumber = sys.argv[1]
        if not Path(sumber).exists():
            sys.exit(f"File tidak ditemukan: {sumber}")
        if Path(sumber).suffix.lower() in EKSTENSI_GAMBAR:
            deteksi_gambar(sumber)
        else:
            deteksi_video(sumber)
