import os
import cv2
import numpy as np

def padding(image, border_width):
    padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_CONSTANT)
    return padded

def crop(image, x_0, x_1, y_0, y_1):
    h, w = image.shape[:2]
    return image[y_0:h - y_1, x_0:w - x_1]

def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def copy(image, emptyPictureArray):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                emptyPictureArray[y, x, ch] = image[y, x, ch]
    return emptyPictureArray

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def hue_shifted(image, emptyPictureArray, hue):
    shifted = (image.astype(np.uint16) + int(hue)) % 256
    emptyPictureArray[:] = shifted.astype(np.uint8)
    return emptyPictureArray

def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        print("Rotation angle must be between 90 and 180 degrees")

def main():
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread("lena.png")

    padded = padding(image, border_width=100)
    cv2.imwrite(os.path.join(output_dir, "padded.jpg"), padded)

    cropped = crop(image, 80, 130, 80, 130)
    cv2.imwrite(os.path.join(output_dir, "cropped.jpg"), cropped)

    resized = resize(image, 200, 200)
    cv2.imwrite(os.path.join(output_dir, "resized.jpg"), resized)

    copylena = np.zeros_like(image, dtype=np.uint8)
    copied = copy(image, copylena)
    cv2.imwrite(os.path.join(output_dir, "copied.jpg"), copied)

    gray = grayscale(image)
    cv2.imwrite(os.path.join(output_dir, "gray.jpg"), gray)

    hsv_img = hsv(image)
    cv2.imwrite(os.path.join(output_dir, "hsv.jpg"), hsv_img)

    shift = np.zeros_like(image, dtype=np.uint8)
    shifted = hue_shifted(image, shift, 50)
    cv2.imwrite(os.path.join(output_dir, "shifted.jpg"), shifted)

    smooth = smoothing(image)
    cv2.imwrite(os.path.join(output_dir, "smoothed.jpg"), smooth)

    rotation90 = rotation(image, 90)
    cv2.imwrite(os.path.join(output_dir, "rotation90.jpg"), rotation90)

    rotation180 = rotation(image, 180)
    cv2.imwrite(os.path.join(output_dir, "rotation180.jpg"), rotation180)

if __name__ == "__main__":
    main()
