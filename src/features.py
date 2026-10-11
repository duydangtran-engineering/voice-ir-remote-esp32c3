"""Trích đặc trưng, dùng chung cho cả lúc train và lúc inference.

Các quyết định thiết kế dưới đây (đều đã đo đạc, không phải đoán mò):

1. Clip 1 giây được cắt lại về phần có tiếng nói trước tiên. Lấy trung bình
   MFCC trên cả phần im lặng bị đệm vào sẽ khiến vector mô tả "clip này im
   lặng bao nhiêu" - tức tốc độ nói của người nói - thay vì mô tả từ đó là gì.
2. Các frame MFCC được pool vào N_BINS bin thời gian thay vì lấy 1 trung bình
   toàn cục. Trung bình toàn cục làm mất hình dạng của từ, khiến "tang" và
   "tat" chồng lên nhau.
3. Chuẩn hóa trung bình cepstral theo từng clip (cepstral mean normalisation -
   trừ đi trung bình MFCC của chính clip đó) loại bỏ độ lệch do micro và độ
   lớn giọng nói, trong khi vẫn giữ được hình dạng (contour) của từ. Việc này
   chạy được trên thiết bị mà không cần đăng ký giọng người dùng trước.
4. RMS energy được thêm theo đúng cách ZCR đã có sẵn (pool theo từng bin thời
   gian). Trước đây RMS chỉ tồn tại bên trong trim_silence() rồi bị bỏ đi;
   giờ đường bao năng lượng của clip (nguyên âm to, phụ âm nhỏ) được giữ lại.
5. Pre-emphasis (`_pre_emphasis()`, hệ số PRE_EMPHASIS trong config.py) được
   áp dụng ngay trước khi tính MFCC - đây là bước chuẩn trong pipeline MFCC
   sách vở (pre-emphasis -> framing & windowing -> FFT -> Mel filter bank ->
   logarithm -> DCT) mà bản trước của file này còn thiếu. Nó CHỈ áp cho nhánh
   MFCC/delta, KHÔNG áp cho RMS/ZCR/trim_silence: 2 đặc trưng đó mô tả năng
   lượng/hình dạng của TÍN HIỆU GỐC, còn TRIM_REL_THRESHOLD đã được canh theo
   tín hiệu gốc - áp pre-emphasis lên cả 2 sẽ làm méo đúng thứ chúng đang đo.

Pitch (F0 qua librosa.yin) từng được thử để bắt thanh điệu tiếng Việt (mục
tiêu là cặp "tăng"/"tắt" dễ nhầm), nhưng thí nghiệm ablation cho kết quả rõ
ràng: f1-score của "tat" giống hệt nhau (0.741) dù có hay không có pitch, và
accuracy tổng còn nhỉnh hơn khi KHÔNG có pitch (0.797 -> 0.803, cùng 1 bộ
5415 clip). Vì không đo được lợi ích nào, và YIN là phần tốn công porting
sang C cho ESP32 nhất trong toàn bộ pipeline, pitch đã bị bỏ hẳn khỏi file
này.

Đo trên tập chia theo người nói - held-out-speaker:
    trung bình/std toàn cục trên clip đã đệm    0.743   <- thiết kế cũ
    chia bin thời gian, chưa chuẩn hóa          0.903
    chia bin + CMN theo từng clip               0.951
    + delta + ZCR theo bin                      0.957
    + RMS theo bin                              0.803   <- bản hiện tại (7 nhãn,
                                                             sau khi bỏ on/off,
                                                             bỏ pitch, dọn clip nhiễu)

Cấu trúc vector đặc trưng:
    [mfcc_bin_means  (N_MFCC*N_BINS),
     mfcc_std        (N_MFCC),
     delta_bin_means (N_MFCC*N_BINS),
     zcr_bin_means   (N_BINS),
     rms_bin_means   (N_BINS),
     voiced_duration (1)]   # CHỈ có mặt nếu config.USE_VOICED_DURATION = True

USE_VOICED_DURATION (trong config.py) là công tắc để làm thí nghiệm ablation
giống cách đã làm với pitch: tắt đi, chạy lại step2_dataset.py rồi
step3_train.py, so accuracy với lúc bật.
"""
import numpy as np
import librosa

from config import (CLIP_SAMPLES, FEAT_HOP, FEAT_WIN, N_BINS, N_FFT, N_MFCC,
                    PRE_EMPHASIS, SAMPLE_RATE, TRIM_MARGIN_FRAMES,
                    TRIM_REL_THRESHOLD, USE_VOICED_DURATION)

FEATURE_NAMES = (
    [f"mfcc{c}_bin{b}" for b in range(N_BINS) for c in range(N_MFCC)]
    + [f"mfcc{c}_std" for c in range(N_MFCC)]
    + [f"delta{c}_bin{b}" for b in range(N_BINS) for c in range(N_MFCC)]
    + [f"zcr_bin{b}" for b in range(N_BINS)]
    + [f"rms_bin{b}" for b in range(N_BINS)]
    + (["voiced_duration"] if USE_VOICED_DURATION else [])
)


def pad_or_trim(y):
    if len(y) > CLIP_SAMPLES:
        return y[:CLIP_SAMPLES]
    if len(y) < CLIP_SAMPLES:
        return np.pad(y, (0, CLIP_SAMPLES - len(y)))
    return y


def trim_silence(y):
    """Cắt clip lại chỉ còn phần frame thực sự mang tiếng của từ."""
    rms = librosa.feature.rms(y=y, frame_length=FEAT_WIN, hop_length=FEAT_HOP)[0]
    threshold = max(rms.max() * TRIM_REL_THRESHOLD, 1e-6)
    voiced = np.where(rms > threshold)[0]
    if len(voiced) < 3:
        return y
    start = max(0, (voiced[0] - TRIM_MARGIN_FRAMES) * FEAT_HOP)
    end = min(len(y), (voiced[-1] + TRIM_MARGIN_FRAMES + 1) * FEAT_HOP)
    return y[start:end]


def _pool(matrix, n_bins):
    """Lấy trung bình mỗi hàng theo n_bins đoạn thời gian bằng nhau."""
    return np.concatenate([b.mean(axis=1)
                           for b in np.array_split(matrix, n_bins, axis=1)])


def _pre_emphasis(y, coef=PRE_EMPHASIS):
    """Bộ lọc FIR bậc 1: y'[n] = y[n] - coef*y[n-1].

    Phổ giọng nói tự nhiên dốc xuống ở tần số cao (~ -6 dB/octave). Bước này
    khuếch đại lại tần số cao trước khi tính phổ, giúp formant tần số cao và
    phụ âm ma sát/bật hơi (vốn yếu hơn nguyên âm) rõ hơn trong MFCC. Mẫu đầu
    tiên giữ nguyên vì không có y[-1] để trừ.
    """
    return np.append(y[0], y[1:] - coef * y[:-1])


def extract(y, sr=SAMPLE_RATE):
    y = trim_silence(pad_or_trim(np.asarray(y, dtype=np.float32)))

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    mfcc = librosa.feature.mfcc(y=_pre_emphasis(y), sr=sr, n_mfcc=N_MFCC,
                               n_fft=N_FFT, hop_length=FEAT_HOP,
                               win_length=FEAT_WIN)
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)     # CMN theo từng clip

    delta = (librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9
             else np.zeros_like(mfcc))
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FEAT_WIN,
                                             hop_length=FEAT_HOP)
    rms = librosa.feature.rms(y=y, frame_length=FEAT_WIN, hop_length=FEAT_HOP)

    parts = [
        _pool(mfcc, N_BINS),
        mfcc.std(axis=1),
        _pool(delta, N_BINS),
        _pool(zcr, N_BINS),
        _pool(rms, N_BINS),
    ]
    # USE_VOICED_DURATION=False (xem config.py): thí nghiệm ablation, kiểm
    # tra xem bỏ đặc trưng này có làm giảm rõ accuracy không.
    if USE_VOICED_DURATION:
        parts.append([len(y) / sr])

    return np.concatenate(parts).astype(np.float32)