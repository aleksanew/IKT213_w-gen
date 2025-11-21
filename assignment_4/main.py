import cv2
import numpy as np
import os

REFERENCE_IMAGE = "reference_img.png"
IMAGE_TO_ALIGN = "align_this.jpg"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def harris(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    out_path = os.path.join(OUTPUT_DIR, "harris_result.png")
    cv2.imwrite(out_path, img)
    return img

def align(image_to_align, reference_image, max_features=10, good_match_percent=0.7):
    img1 = cv2.imread(image_to_align)
    img2 = cv2.imread(reference_image)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)  # image_to_align
    kp2, des2 = sift.detectAndCompute(gray2, None)  # reference

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    print(f"Total matches: {len(matches)}")

    good_matches = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good_matches.append(m)

    print(f"Good matches: {len(good_matches)}")

    good_matches = sorted(good_matches, key = lambda x: x.distance)[:max_features]

    print(f"Matches after good_match_percent: {len(good_matches)}")

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    matched_path = os.path.join(OUTPUT_DIR, "matched_image.png")
    cv2.imwrite(matched_path, matched_img)

    if len(good_matches) < 4:
        print(f"Not enough matches: ({len(good_matches)})")
        return None, matched_img

    p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC,  5.0)

    height, width = gray2.shape
    im2_aligned = cv2.warpPerspective(img1, H, (width, height))

    aligned_path = os.path.join(OUTPUT_DIR, "aligned_image.png")
    cv2.imwrite(aligned_path, im2_aligned)

    return im2_aligned, matched_img

if __name__ == "__main__":
    harris(REFERENCE_IMAGE)
    align(IMAGE_TO_ALIGN, REFERENCE_IMAGE, max_features=10, good_match_percent=0.7)