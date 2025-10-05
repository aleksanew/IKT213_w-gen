import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def mb(x):
    return x / (1024.0 * 1024.0)

def preprocess_photo(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def preprocess_fp_orb(path):
    g = cv2.imread(path, 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw

def preprocess_fp_sift(path):
    g = cv2.imread(path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def plot_confmat(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["No Match (0)", "Match (1)"])
    ax.set_yticklabels(["Match (0)", "Match (1)"])
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > 0.5*vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()
    return cm

def plot_match_counts(folder_names, counts, threshold, title, save_path=None):
    colors = ["green" if "same" in f.lower() else "red" for f in folder_names]
    x = np.arange(len(folder_names))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.bar(x, counts, color=colors)
    ax.axhline(threshold, linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.set_ylabel("Number of good matches")
    ax.set_title(title + " (Green=Same, Red=Different)")
    ax.set_xticks(x)
    ax.set_xticklabels(folder_names, rotation=60, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    plt.show()

def detect_and_match(method, g1, g2):
    if method == "SIFT":
        sift = cv2.SIFT_create(nfeatures=4000)
        t0 = time.time()
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)
        t1 = time.time()
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=100))
        matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        t2 = time.time()
        desc_bytes = (des1.nbytes if des1 is not None else 0) + (des2.nbytes if des2 is not None else 0)
    else:  # ORB
        orb = cv2.ORB_create(nfeatures=4000)
        t0 = time.time()
        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        t1 = time.time()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        t2 = time.time()
        desc_bytes = (des1.nbytes if des1 is not None else 0) + (des2.nbytes if des2 is not None else 0)

    good = [m for m, n in matches if m.distance < 0.65 * n.distance]
    stats = dict(
        kp1=len(kp1), kp2=len(kp2), good=len(good),
        detect_ms=(t1 - t0)*1000.0, match_ms=(t2 - t1)*1000.0,
        total_ms=(t2 - t0)*1000.0, desc_MB=mb(desc_bytes)
    )
    return kp1, kp2, good, stats

def run_uia(img1_path, img2_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    g1 = preprocess_photo(img1)
    g2 = preprocess_photo(img2)

    results = []
    for method in ["ORB", "SIFT"]:
        kp1, kp2, good, s = detect_and_match(method, g1, g2)
        vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        out_path = os.path.join(out_dir, f"uia_{method.lower()}_matches.png")
        cv2.imwrite(out_path, vis)
        s.update(method=method, out=out_path)
        results.append(s)
        print(f"{method:>4s} (UiA)  kp1={s['kp1']}  kp2={s['kp2']}  good={s['good']}  "
              f"desc={s['desc_MB']:.2f} MB  time(ms) detect={s['detect_ms']:.1f} match={s['match_ms']:.1f} total={s['total_ms']:.1f}  -> {out_path}")

    print("\nUiA summary (bigger ‘good’ usually means better):")
    for s in results:
        print(f"  {s['method']}: good={s['good']}, kp1={s['kp1']}, kp2={s['kp2']}, desc={s['desc_MB']:.2f} MB, total_ms={s['total_ms']:.1f}")
    return results

def run_fingerprints(dataset_path, out_dir, method="ORB", threshold=20):
    os.makedirs(out_dir, exist_ok=True)
    y_true, y_pred = [], []
    folder_names, counts = [], []

    detect_ms_list, match_ms_list, total_ms_list = [], [], []
    kp_list, desc_mb_list = [], []

    folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,f))])
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".tif")])
        if len(files) != 2:
            continue
        p1 = os.path.join(folder_path, files[0])
        p2 = os.path.join(folder_path, files[1])

        if method == "SIFT":
            g1 = preprocess_fp_sift(p1)
            g2 = preprocess_fp_sift(p2)
        else:
            g1 = preprocess_fp_orb(p1)
            g2 = preprocess_fp_orb(p2)

        kp1, kp2, good, s = detect_and_match(method, g1, g2)
        vis = cv2.drawMatches(g1, kp1, g2, kp2, good, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        tag = "matched" if len(good) > threshold else "unmatched"
        cv2.imwrite(os.path.join(out_dir, f"{folder}_{method.lower()}_{tag}.png"), vis)

        folder_names.append(folder)
        counts.append(len(good))

        actual_same = 1 if "same" in folder.lower() else 0
        y_true.append(actual_same)
        y_pred.append(1 if len(good) > threshold else 0)

        detect_ms_list.append(s['detect_ms'])
        match_ms_list.append(s['match_ms'])
        total_ms_list.append(s['total_ms'])
        kp_list.append(0.5*(s['kp1']+s['kp2']))
        desc_mb_list.append(s['desc_MB'])

        print(f"{folder:>12s} [{method}]  good={len(good):3d}  -> {tag.upper()}  "
              f"time(ms) detect={s['detect_ms']:.1f} match={s['match_ms']:.1f} total={s['total_ms']:.1f}  desc={s['desc_MB']:.2f} MB")

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp+tn+fp+fn))
    print(f"\nData_Check Accuracy [{method}]: {acc*100:.2f}% ({tp+tn}/{tp+tn+fp+fn})")

    plot_confmat(y_true, y_pred, title=f"Confusion Matrix {method}",
                 save_path=os.path.join(out_dir, f"confmat_{method.lower()}.png"))
    plot_match_counts(folder_names, counts, threshold,
                      title=f"Match Counts per Folder ({method})",
                      save_path=os.path.join(out_dir, f"match_counts_{method.lower()}.png"))

    # resource + speed summaries
    print(f"\n[{method}] resources/speed over data_check:")
    print(f"  avg keypoints per image ~ {np.mean(kp_list):.0f}")
    print(f"  avg descriptor memory   ~ {np.mean(desc_mb_list):.2f} MB (both images combined)")
    print(f"  avg times (ms): detect={np.mean(detect_ms_list):.1f}, match={np.mean(match_ms_list):.1f}, total={np.mean(total_ms_list):.1f}")
    return dict(acc=acc, avg_kp=np.mean(kp_list), avg_desc_mb=np.mean(desc_mb_list),
                avg_detect_ms=np.mean(detect_ms_list), avg_match_ms=np.mean(match_ms_list),
                avg_total_ms=np.mean(total_ms_list))

if __name__ == "__main__":
    img1_path = r"C:\Users\aew1\IKT213_w-gen\Figureprint\UiA front1.png"
    img2_path = r"C:\Users\aew1\IKT213_w-gen\Figureprint\UiA front3.jpg"
    out_dir   = r"C:\Users\aew1\IKT213_w-gen\Figureprint\results"

    print("\n=== UiA images ===")
    uia_stats = run_uia(img1_path, img2_path, out_dir)

    dataset_path = r"C:\Users\aew1\IKT213_w-gen\Figureprint\dataset_folder\data_check"
    print("\n=== Fingerprints: ORB pipeline ===")
    orb_stats  = run_fingerprints(dataset_path, out_dir, method="ORB",  threshold=20)

    print("\n=== Fingerprints: SIFT pipeline ===")
    sift_stats = run_fingerprints(dataset_path, out_dir, method="SIFT", threshold=20)

    print("\n=== Summary (headline numbers for report) ===")
    for s in uia_stats:
        print(f"UiA {s['method']:>4s} -> good={s['good']}, kp≈{(s['kp1']+s['kp2'])//2}, "
              f"desc={s['desc_MB']:.2f} MB, total_ms={s['total_ms']:.1f}")

    print(f"\nFingerprints ORB  -> acc={orb_stats['acc']*100:.1f}%, kp~{orb_stats['avg_kp']:.0f}, "
          f"desc={orb_stats['avg_desc_mb']:.2f} MB, total_ms={orb_stats['avg_total_ms']:.1f}")
    print(f"Fingerprints SIFT -> acc={sift_stats['acc']*100:.1f}%, kp~{sift_stats['avg_kp']:.0f}, "
          f"desc={sift_stats['avg_desc_mb']:.2f} MB, total_ms={sift_stats['avg_total_ms']:.1f}")
