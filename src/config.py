"""Cấu hình dùng chung cho mọi bước trong pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"          # file .m4a gốc, mỗi nhãn 1 thư mục
CLIPS_DIR = ROOT / "data" / "clips"      # đoạn .wav 1 giây sau khi cắt
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Audio
SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)

# Tách từ dựa trên năng lượng (energy based word splitting)
FRAME_LENGTH = 512
HOP_LENGTH = 128
ENERGY_PERCENTILE = 75      # ngưỡng = percentile của RMS từng frame
MIN_WORD_MS = 150           # ngắn hơn mức này -> đánh dấu "_SHORT", KHÔNG tự xoá
MIN_SILENCE_MS = 120        # khoảng lặng cần có để kết thúc 1 từ
MAX_WORD_MS = 1000          # dài hơn mức này = có thể 2 từ dính liền nhau ->
                              # đánh dấu "_LONG", KHÔNG tự xoá (xem step1_split.py)
PAD_MS = 60                 # đệm thêm quanh mỗi từ phát hiện được

# Override riêng cho từng bản ghi, key dạng "<nhãn>/<tên file>". Có người nói
# nhanh hơn, nghỉ ngắn hơn người khác, nên 1 ngưỡng chung không hợp với tất cả.
# Dùng step1c_validate.py để tìm người cần override:
#   LONG  -> giảm min_silence_ms (các từ đang bị dính vào nhau)
#   SHORT -> tăng min_silence_ms, hoặc giảm energy_percentile
SPLIT_OVERRIDES = {
    "tang/nhi_01": {"min_silence_ms": 70},
}

# Trích đặc trưng
N_MFCC = 13
N_FFT = 512
FEAT_HOP = 160              # bước nhảy 10 ms
FEAT_WIN = 400               # cửa sổ 25 ms
N_BINS = 10                  # số bin thời gian mà MFCC/RMS/ZCR/pitch được pool vào
TRIM_REL_THRESHOLD = 0.06    # RMS của frame dưới tỉ lệ này so với đỉnh = im lặng
TRIM_MARGIN_FRAMES = 6       # số frame giữ lại ở 2 đầu vùng có tiếng

# Pre-emphasis: bộ lọc FIR bậc 1 y'[n] = y[n] - PRE_EMPHASIS*y[n-1], khuếch đại
# tần số cao trước khi tính MFCC (xem features.py). 0.97 là giá trị kinh điển
# trong nhận dạng giọng nói (Rabiner & Schafer), hiếm khi cần chỉnh.
PRE_EMPHASIS = 0.97

# Pitch (F0/YIN) đã bị bỏ hẳn: thí nghiệm ablation cho thấy f1-score của "tat"
# - đúng từ mà pitch được thêm vào để cứu - giống hệt nhau dù có hay không có
# pitch (0.741 cả 2 lần), còn accuracy tổng thậm chí nhỉnh hơn khi KHÔNG có
# pitch (0.797 -> 0.803). Không đáng công port thuật toán YIN sang C cho
# ESP32 để đổi lấy 0 lợi ích đo được. Xem features.py để biết cấu trúc vector
# đặc trưng hiện tại.

# Công tắc bật/tắt voiced_duration (độ dài phần có tiếng của clip, tính bằng
# giây) để làm thí nghiệm ablation tương tự - xem features.py.
USE_VOICED_DURATION = True

# Các lớp âm học mà mô hình thực sự học. Giữ các từ đồng nghĩa tách riêng ở đây:
# "on" và "bat" nghe hoàn toàn khác nhau nên phải là 2 lớp riêng biệt.
# "noise" là người nói chuyện bình thường không ra lệnh, cộng thêm các từ gần
# giống lệnh. "silence" chỉ là tiếng nền của phòng - quạt, tiếng xe, phòng
# trống. Hai lớp này tách riêng vì nghe hoàn toàn khác nhau, và chỉ "silence"
# mới an toàn để trộn vào clip lệnh khi augment.
LABELS = ["tang", "giam", "bat", "tat", "on", "off", "mode", "noise", "silence"]

# Từ đồng nghĩa được gộp ở tầng hành động (action), không gộp trong model.
# Firmware dùng chung bảng này, nên thêm từ đồng nghĩa không bao giờ cần train lại.
ACTION_MAP = {
    "tang": "VOLUME_UP",
    "giam": "VOLUME_DOWN",
    "bat": "POWER_ON",
    "on": "POWER_ON",
    "tat": "POWER_OFF",
    "off": "POWER_OFF",
    "mode": "MODE_SWITCH",
    "noise": "IGNORE",
    "silence": "IGNORE",
}

RANDOM_SEED = 42