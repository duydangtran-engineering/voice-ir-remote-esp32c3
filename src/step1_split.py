"""Bước 1: cắt các bản ghi .m4a dài thành các clip .wav 1 từ, dài 1 giây.

Cấu trúc thư mục đầu vào mong muốn:
    data/raw/tang/vinh_01.m4a
    data/raw/tang/nam_01.m4a
    data/raw/giam/...
    data/raw/noise/...      (tạp âm nền + nói chuyện linh tinh, chặt cơ học
                             thành khúc 1 giây chồng lấp - xem split_mechanical())

Đầu ra:
    data/clips/tang/vinh_01_000.wav ...
    data/clips/tang/vinh_01_003_LONG.wav   (đoạn dài bất thường - có thể 2 từ dính liền)
    data/clips/tang/vinh_01_007_SHORT.wav  (đoạn ngắn bất thường - có thể tiếng ồn/gió)

Đoạn "LONG"/"SHORT" KHÔNG bị tự động xoá - bản trước của file này tự xoá thẳng
đoạn quá dài (MAX_WORD_MS), điều đó nghĩa là 1 quyết định xoá dữ liệu được
đưa ra chỉ dựa vào 1 con số ngưỡng, không ai nghe lại để xác nhận. Giống bài
học từ step1d_flag_noisy.py (xoá cả loạt 150 clip theo ngưỡng số làm hanh bị
tệ đi), cách an toàn hơn là đánh dấu ngay trong TÊN FILE để dễ lọc/nghe lại,
rồi tự tay xoá file nào đúng là lỗi thật.
"""
import sys

import librosa
import numpy as np
import soundfile as sf

import audio_io
from config import (CLIPS_DIR, CLIP_SAMPLES, ENERGY_PERCENTILE, FRAME_LENGTH,
                    HOP_LENGTH, LABELS, MAX_WORD_MS, MIN_SILENCE_MS,
                    MIN_WORD_MS, PAD_MS,
                    RAW_DIR, SAMPLE_RATE, SPLIT_OVERRIDES)

AUDIO_EXT = {".m4a", ".mp4", ".wav", ".mp3", ".aac", ".ogg", ".flac"}


def detect_words(y, energy_percentile=ENERGY_PERCENTILE,
                 min_word_ms=MIN_WORD_MS, min_silence_ms=MIN_SILENCE_MS,
                 pad_ms=PAD_MS, max_word_ms=MAX_WORD_MS):
    """Trả về danh sách (start, end, tag) - chỉ số sample cho mỗi từ được nói.

    tag là "" (bình thường), "LONG" (dài hơn max_word_ms - có thể 2 từ dính
    liền nhau) hoặc "SHORT" (ngắn hơn min_word_ms - có thể tiếng ồn/gió ngắn
    bị hiểu nhầm thành từ). Cả 2 loại bất thường đều được GIỮ LẠI, không tự
    xoá - xem process_label() để biết cách đánh dấu vào tên file.
    """
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH,
                              hop_length=HOP_LENGTH)[0]
    threshold = np.percentile(rms, energy_percentile) * 0.35
    threshold = max(threshold, rms.max() * 0.08)
    active = rms > threshold

    min_sil = int(min_silence_ms / 1000 * SAMPLE_RATE / HOP_LENGTH)

    raw = []    # (start_frame, end_frame), chưa lọc theo độ dài
    start = None
    silence = 0
    for i, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_sil:
                raw.append((start, i - silence))
                start = None
                silence = 0
    if start is not None:
        raw.append((start, len(active)))

    pad = int(pad_ms / 1000 * SAMPLE_RATE)
    min_dur = min_word_ms / 1000 * SAMPLE_RATE
    max_dur = max_word_ms / 1000 * SAMPLE_RATE
    out = []
    for s, e in raw:
        s = max(0, s * HOP_LENGTH - pad)
        e = min(len(y), e * HOP_LENGTH + pad)
        dur = e - s
        if dur > max_dur:
            tag = "LONG"
        elif dur < min_dur:
            tag = "SHORT"
        else:
            tag = ""
        out.append((s, e, tag))
    return out


def to_fixed_length(chunk):
    """Căn giữa 1 đoạn có độ dài thay đổi vào trong buffer 1 giây."""
    if len(chunk) >= CLIP_SAMPLES:
        centre = len(chunk) // 2
        half = CLIP_SAMPLES // 2
        return chunk[max(0, centre - half): max(0, centre - half) + CLIP_SAMPLES]
    buf = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    offset = (CLIP_SAMPLES - len(chunk)) // 2
    buf[offset:offset + len(chunk)] = chunk
    return buf


def split_mechanical(y):
    """Dùng cho "noise" VÀ "silence": chặt thành các cửa sổ 1 giây chồng lấp
    50%, không tìm khoảng lặng, không cần bảo vệ ranh giới sự kiện nào cả.

    Từng có 2 hàm riêng: split_noise() cắt ở khoảng lặng (coi nội dung là
    các "sự kiện" rời rạc như từ lệnh, phải tách riêng từng sự kiện) và
    split_silence() chặt cơ học (vì tiếng nền phòng đồng nhất, không có sự
    kiện gì để bảo vệ).

    split_noise() sinh ra hàng loạt bug (2 từ dính 1 clip do thuật toán tìm
    điểm cắt "đẹp" gần mốc 1 giây bỏ sót khoảng lặng gần đầu đoạn) vì nó giả
    định noise luôn là các đoạn nói rời rạc, cách nhau bởi khoảng lặng rõ -
    giả định này sai với phần lớn nội dung noise thực tế (nói chuyện tự
    nhiên, liên tục, gần như không dừng - xem README phần thu âm noise).

    Câu hỏi đúng không phải "noise cắt giống từ lệnh hay giống silence", mà
    là: model có cần mỗi clip noise chỉ chứa ĐÚNG 1 đơn vị âm thanh sạch,
    không lẫn lộn, như với từ lệnh không? Không - model chỉ cần học "đây
    KHÔNG phải 1 trong 7 lệnh", không cần phân biệt hình dạng từng âm trong
    noise với nhau. Vì vậy noise không cần bảo vệ ranh giới sự kiện gì cả,
    y hệt silence - 1 clip chứa nửa câu hay dính 2 âm đệm hoàn toàn không
    sao, thậm chí còn thực tế hơn vì live_demo.py cũng bắt 1 đoạn 1 giây bất
    kỳ, không quan tâm ranh giới từ. Gộp về 1 hàm chặt cơ học duy nhất, bỏ
    hẳn LONG/SHORT cho 2 nhãn này - không còn gì để đánh dấu vì không còn
    "sự kiện" nào có thể bị cắt sai cả.
    """
    step = CLIP_SAMPLES // 2
    chunks = [y[i:i + CLIP_SAMPLES]
             for i in range(0, max(1, len(y) - CLIP_SAMPLES + 1), step)]
    if not chunks:
        chunks = [y]
    return [(c, "") for c in chunks]


def process_label(label):
    src_dir = RAW_DIR / label
    dst_dir = CLIPS_DIR / label
    if not src_dir.exists():
        print(f"[bỏ qua] không tìm thấy {src_dir}")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for path in sorted(src_dir.iterdir()):
        if path.suffix.lower() not in AUDIO_EXT:
            continue
        y = audio_io.load(path, sr=SAMPLE_RATE)
        y = librosa.util.normalize(y)

        if label in ("noise", "silence"):
            chunks = split_mechanical(y)
        else:
            opts = SPLIT_OVERRIDES.get(f"{label}/{path.stem}", {})
            if opts:
                print(f"    (override: {opts})")
            segments = detect_words(y, **opts)
            chunks = [(to_fixed_length(y[s:e]), tag) for s, e, tag in segments]

        n_long = n_short = 0
        for idx, (chunk, tag) in enumerate(chunks):
            suffix = f"_{tag}" if tag else ""
            out = dst_dir / f"{path.stem}_{idx:03d}{suffix}.wav"
            sf.write(out, to_fixed_length(np.asarray(chunk)), SAMPLE_RATE,
                     subtype="PCM_16")
            n_long += tag == "LONG"
            n_short += tag == "SHORT"

        flags = []
        if n_long:
            flags.append(f"{n_long} _LONG")
        if n_short:
            flags.append(f"{n_short} _SHORT")
        note = f"  (⚠ {', '.join(flags)} - nghe thử trước khi train)" if flags else ""
        print(f"  {path.name}: {len(chunks)} clip{note}")
        total += len(chunks)
    print(f"[{label}] {total} clip")
    return total


def main():
    labels = sys.argv[1:] or LABELS
    grand = sum(process_label(label) for label in labels)
    print(f"\nTổng số clip: {grand}")
    print(f"Nghe lại bằng tai trong {CLIPS_DIR} trước khi train.")
    print("File tên có _LONG/_SHORT là đoạn bất thường (2 từ dính liền / có thể "
         "là tiếng ồn) - ưu tiên nghe thử các file này trước, xoá tay nếu đúng "
         "là lỗi (giữ nguyên nếu vẫn là 1 từ nói bình thường, chỉ hơi dài/ngắn).")


if __name__ == "__main__":
    main()