# Voice → IR: ML pipeline (tang / giam / bat / tat / noise)

## 1. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`librosa` đọc `.m4a` qua `audioread` → cần **ffmpeg** trong PATH
(`winget install ffmpeg` hoặc `choco install ffmpeg`).

## 2. Bỏ file thu âm vào đúng thư mục

```
data/raw/tang/vinh_01.m4a      <- tên file: <tên người>_<lần thu>.m4a
data/raw/tang/nam_01.m4a
data/raw/giam/...
data/raw/bat/...
data/raw/tat/...
data/raw/noise/vinh_01.m4a     <- tạp âm: quạt, TV, nói chuyện, im lặng
```

Tên người ở đầu file là BẮT BUỘC — script dùng nó để chia train/test theo
người nói (speaker-independent), đây mới là con số accuracy trung thực.

## 3. Chạy 4 bước

```bash
cd src
python step1_split.py      # cắt m4a dài -> clip 1 giây
python step2_dataset.py    # trích MFCC + delta + ZCR + RMS -> dataset.npz
python step3_train.py      # train + so sánh RandomForest vs SVM
python step3b_report.py <ten> #train từng người
python step4_export.py     # xuất models/voice_model.h cho ESP32
```
Get-ChildItem -Path .\data\clips -Recurse -Include *_LONG*.wav,*_SHORT*.wav | Remove-Item
>dùng để xóa mấy file bị báo hiệu long or short

Test nhanh 1 file bất kỳ: `python predict.py ../data/raw/tang/vinh_02.m4a`

> **Sửa `features.py`/`config.py` (đổi `USE_VOICED_DURATION` hoặc bất kỳ đặc trưng
> nào)?** Bắt buộc chạy lại `step2_dataset.py` rồi `step3_train.py` —
> `dataset.npz` và `model.joblib` cũ được trích theo vector đặc trưng cũ nên
> không tương thích với code `features.py` mới (số chiều lệch nhau).

## 4. Sau step1 — PHẢI nghe lại

Mở `data/clips/<label>/` nghe vài chục file. Nếu bị cắt cụt từ hoặc dính 2 từ
vào 1 clip, chỉnh trong `config.py`:

| Vấn đề | Sửa |
|---|---|
| Cắt sót từ (ít clip quá) | giảm `ENERGY_PERCENTILE` (75 → 60) |
| Bắt cả tiếng ồn (nhiều clip rác) | tăng `ENERGY_PERCENTILE` (75 → 85) |
| 2 từ dính vào 1 clip | giảm `MIN_SILENCE_MS` (120 → 80) và nói cách xa hơn |
| 1 từ bị tách làm 2 | tăng `MIN_SILENCE_MS` (120 → 200) |
| Mất phụ âm đầu/cuối | tăng `PAD_MS` (60 → 100) |

Cách thu để cắt dễ: nói từ đó lặp lại, **nghỉ rõ ràng ~0.5 giây giữa mỗi lần**,
mỗi file ~20 lần. Mục tiêu ≥ 200 clip/lớp, ít nhất 4–5 người, có cả file thu
trong môi trường ồn.

### 4b. (tùy chọn) Lọc bớt clip là tiếng ồn/gió bị cắt nhầm thành "từ"

Bộ tách từ ở step1 dựa trên năng lượng (RMS), nên 1 cơn gió thổi vào micro
hoặc tiếng "phù" hơi cũng đủ to để bị hiểu nhầm là 1 từ và bị cắt ra thành
clip — những clip này là nhiễu thật sự trong tập train. Nghe thủ công vài
nghìn clip để tìm chúng rất mất công, nên có `step1d_flag_noisy.py` để
xếp hạng trước:

```bash
python step1d_flag_noisy.py          # xếp hạng, copy ~150 clip nghi ngờ nhất
                                      # ra data/clips_review/ (không đụng bản gốc)
#  ... nghe qua data/clips_review/, xoá file nào đúng là gió/ồn ngay trong đó ...
python step1d_flag_noisy.py --apply  # đồng bộ: xoá đúng những file đó khỏi data/clips
```

Script chỉ xếp hạng theo độ "giống tiếng ồn" (phổ phẳng, đường bao năng lượng
không có đỉnh rõ, kéo dài gần hết clip) — **không tự xoá gì**, vì chỉ dựa vào
vài chỉ số thống kê sẽ có trường hợp đoán nhầm (ví dụ 1 từ nói to/gằn giọng có
thể bị chấm điểm nghi ngờ cao dù là giọng thật). Luôn nghe lại trước khi xoá.

## 5. Đọc kết quả (phần "phân tích mô hình" của đề bài)

`step3_train.py` in ra precision/recall/f1 từng lớp và lưu confusion matrix vào
`reports/confusion_*.png`. Hai điều gần như chắc chắn sẽ thấy:

- **tang ↔ tat** dễ nhầm nhất (cùng nguyên âm /a/, chỉ khác thanh điệu và phụ âm cuối).
  Đã thử thêm đặc trưng pitch contour (`librosa.yin`, chuẩn hoá log2 theo từng
  clip) để bắt thông tin thanh điệu, nhưng thí nghiệm ablation (bật/tắt pitch,
  chạy lại trên cùng 5415 clip) cho thấy f1-score của "tat" y hệt nhau (0.741)
  dù có hay không có pitch, và accuracy tổng còn nhỉnh hơn khi bỏ pitch
  (0.797 → 0.803). Kết luận: pitch không đóng góp gì đo được cho đúng cặp từ
  nó được thêm vào để cứu, nên đã bỏ hẳn khỏi `features.py` — đỡ phải port
  YIN (tốn CPU nhất trong pipeline) sang C cho ESP32.
- **noise** thường có recall cao nhất — đó là lớp dễ, đừng lấy nó để khoe accuracy tổng.

Random Forest gần như luôn thắng ở đây vì `emlearn` xuất cây quyết định ra C rất
gọn, còn SVM RBF phải mang theo support vectors → nặng cho ESP32-C3.

Sau khi train xong, xem khối "Top features" / bảng "Tổng hợp theo nhóm đặc
trưng" cuối log của `step3_train.py` (chỉ in khi Random Forest thắng) để
quyết định nhóm nào thực sự đáng port sang C. Cách làm đã dùng cho pitch:
bật/tắt bằng công tắc `USE_*` trong `config.py`, chạy lại step2+step3, so
accuracy 2 lần — nếu bỏ đi mà accuracy gần như không đổi (đặc biệt ở đúng
lớp mà đặc trưng đó nhắm tới), bỏ hẳn nhóm đó khi triển khai lên ESP32-C3.
`USE_VOICED_DURATION` đang được thử theo đúng cách này.

## 6. Nối sang phần nhúng

`models/voice_model.h` chỉ chứa **bộ phân lớp**. Trên ESP32 bạn vẫn phải tự viết
lại phần trích đặc trưng bằng C với ĐÚNG tham số trong `config.py`
(16 kHz, n_mfcc=13, n_fft=512, hop=160, win=400) và đúng thứ tự vector ghi ở cuối
file .h. Sai thứ tự hoặc sai tham số → model chạy được nhưng đoán bừa.

Cần thêm trên ESP32: RMS (dễ, chỉ là căn bậc hai trung bình bình phương từng
frame — rẻ) và ZCR. Pitch (YIN) đã bị bỏ hẳn — xem mục 5 để biết lý do (thí
nghiệm ablation không thấy lợi ích đo được, mà YIN là phần tốn CPU nhất nếu
phải port).

Kiểm chứng: in vector đặc trưng của 1 clip ở Python và cùng clip đó trên ESP32,
2 dãy số phải khớp tới ~2 chữ số thập phân trước khi tin vào kết quả predict.