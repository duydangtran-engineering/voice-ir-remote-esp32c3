"""Bước 3: train và so sánh Random Forest với SVM, rồi lưu model tốt hơn.

Các quy tắc đánh giá giúp con số đáng tin cậy:

1. Held-out speaker (giữ riêng 1 người nói ra ngoài). Mỗi fold test trên 1
   giọng hoàn toàn vắng mặt lúc train, nên điểm số phản ánh đúng trải nghiệm
   của 1 người lạ dùng thiết bị - chứ không phải những người đã thu dữ liệu.
2. Clip augment chỉ dùng để train, không bao giờ dùng để test. Một bản sao
   trộn tạp âm của 1 clip không phải là bằng chứng mới: chấm điểm trên nó sẽ
   hoặc làm đẹp điểm số (vì bản gốc sạch của nó đã có trong tập train) hoặc
   phạt oan model vì 1 đề thi khó hơn. Fold test chỉ chứa bản ghi gốc, nên
   điểm số vẫn so sánh được dù có chạy step1b_augment.py hay không.
"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import ACTION_MAP, LABELS, MODEL_DIR, RANDOM_SEED, REPORT_DIR

MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
    "svm_rbf": make_pipeline(
        StandardScaler(),
        SVC(C=10, gamma="scale", kernel="rbf", class_weight="balanced",
            random_state=RANDOM_SEED)),
}


def feature_group(name):
    """Nhóm 1 tên đặc trưng (vd 'mfcc3_bin7') về nhóm gốc của nó ('mfcc_bin').

    Cần cái này vì nhìn top 10 đặc trưng riêng lẻ dễ đánh giá sai: 1 nhóm có
    nhiều bin (mfcc_bin có 130) rất dễ có vài bin lọt top 10 dù cả nhóm không
    thật sự nổi trội hơn 1 nhóm ít bin (rms_bin chỉ có 10) mà mỗi bin đều
    đóng góp đều tay nhưng không có bin nào đủ nổi để vào top 10.
    """
    if name == "voiced_duration":
        return "voiced_duration"
    if name.startswith("mfcc") and name.endswith("_std"):
        return "mfcc_std"
    if name.startswith("mfcc"):
        return "mfcc_bin"
    if name.startswith("delta"):
        return "delta_bin"
    if name.startswith("zcr_bin"):
        return "zcr_bin"
    if name.startswith("rms_bin"):
        return "rms_bin"
    return "khac"


def plot_confusion(y_true, y_pred, labels, name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d",
        xticks_rotation=45)
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_DIR / f"confusion_{name}.png", dpi=150)
    plt.close(fig)


def speaker_folds(speakers, is_aug):
    """Mỗi fold ứng với 1 người nói: train trên mọi người khác, test trên clip gốc của họ."""
    for held_out in sorted(set(speakers)):
        train = np.where(speakers != held_out)[0]
        test = np.where((speakers == held_out) & ~is_aug)[0]
        if len(test):
            yield held_out, train, test


def evaluate(model, X, y, speakers, is_aug):
    y_true, y_pred, per_speaker = [], [], {}
    for name, train, test in speaker_folds(speakers, is_aug):
        fitted = clone(model).fit(X[train], y[train])
        pred = fitted.predict(X[test])
        per_speaker[name] = accuracy_score(y[test], pred)
        y_true.append(y[test])
        y_pred.append(pred)
    return np.concatenate(y_true), np.concatenate(y_pred), per_speaker


def main():
    data = np.load(MODEL_DIR / "dataset.npz", allow_pickle=True)
    X, y, speakers = data["X"], data["y"], data["speakers"]
    is_aug = data["is_aug"] if "is_aug" in data else np.zeros(len(y), bool)
    labels = [l for l in LABELS if l in set(y)]

    print(f"{len(y)} clip, {(~is_aug).sum()} gốc, {is_aug.sum()} augment")
    print(f"người nói: {', '.join(sorted(set(speakers)))}\n")

    scores = {}
    for name, model in MODELS.items():
        y_true, y_pred, per_speaker = evaluate(model, X, y, speakers, is_aug)
        acc = accuracy_score(y_true, y_pred)
        scores[name] = acc

        print(f"===== {name} | accuracy held-out-speaker = {acc:.3f} =====")
        print("  " + "  ".join(f"{s}={v:.3f}" for s, v in per_speaker.items()))
        print(classification_report(y_true, y_pred, labels=labels, digits=3))
        plot_confusion(y_true, y_pred, labels, name)

        words = y_true != "noise"
        print(f"--- chỉ tính lệnh ({words.sum()} clip) = "
              f"{accuracy_score(y_true[words], y_pred[words]):.3f}")

        # Nhầm "on" thành "bat" vẫn kích đúng mã IR, nên đây mới là con số
        # thiết bị thực sự làm đúng.
        act_true = np.array([ACTION_MAP.get(v, "IGNORE") for v in y_true])
        act_pred = np.array([ACTION_MAP.get(v, "IGNORE") for v in y_pred])
        print(f"--- action level = {accuracy_score(act_true, act_pred):.3f}\n")
        plot_confusion(act_true, act_pred, sorted(set(act_true)),
                       f"{name}_action")

    best = max(scores, key=scores.get)
    print(f"Model tốt nhất: {best} ({scores[best]:.3f})")

    final = clone(MODELS[best]).fit(X, y)
    joblib.dump({"model": final, "labels": labels}, MODEL_DIR / "model.joblib")
    print(f"Đã lưu -> {MODEL_DIR / 'model.joblib'}")

    if best == "random_forest":
        names = data["feature_names"]
        importances = final.feature_importances_

        order = np.argsort(importances)[::-1][:10]
        print("\nĐặc trưng quan trọng nhất (10 bin riêng lẻ cao nhất):")
        for i in order:
            print(f"  {names[i]:16s} {importances[i]:.4f}")

        # Tổng hợp theo NHÓM (mfcc_bin, delta_bin, zcr_bin, rms_bin,
        # mfcc_std, voiced_duration) - câu trả lời đáng tin hơn cho câu hỏi
        # "nhóm đặc trưng nào thực sự đáng giữ", vì 1 nhóm ít bin (rms_bin:
        # 10 bin) rất dễ bị nhìn nhầm là "yếu" nếu chỉ so từng bin lẻ với 1
        # nhóm nhiều bin (mfcc_bin: 130 bin).
        groups = {}
        for name, imp in zip(names, importances):
            g = feature_group(str(name))
            gtotal, gcount = groups.get(g, (0.0, 0))
            groups[g] = (gtotal + imp, gcount + 1)

        print("\nTổng hợp theo nhóm đặc trưng (tổng importance phải ≈ 1.0):")
        print(f"  {'nhóm':16s} {'số bin':>7} {'tổng':>8} {'%':>7} {'TB/bin':>8}")
        for g, (gtotal, gcount) in sorted(groups.items(), key=lambda kv: -kv[1][0]):
            pct = 100 * gtotal / importances.sum()
            print(f"  {g:16s} {gcount:7d} {gtotal:8.4f} {pct:6.1f}% {gtotal/gcount:8.4f}")
        print("\n'tổng' = nhóm đó đóng góp bao nhiêu % sức nặng của cả model.")
        print("'TB/bin' = trung bình mỗi bin trong nhóm đó đóng góp bao nhiêu -")
        print("so cột này giữa các nhóm mới công bằng (không bị số lượng bin")
        print("chi phối) - nhóm nào TB/bin thấp hơn hẳn phần còn lại mới thực")
        print("sự đáng cân nhắc bỏ, không phải nhóm nào tổng thấp vì ít bin.")


if __name__ == "__main__":
    main()