import numpy as np
import cv2
import os

def saveimage(image,filename):
    cv2.imwrite(filename,image)

def turngray(image):
    turntogray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return turntogray

def sobel_edge_detection(image):
    gray = turngray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=1)
    abs_sobel = np.absolute(sobel)

    if abs_sobel.max() > 0:
        sobel_scaled = np.uint8(255 * abs_sobel / abs_sobel.max())
    else:
        sobel_scaled = np.uint8(abs_sobel)

    return sobel_scaled

def canny_edge_detection(image, threshold_1=50, threshold_2=50):
    gray = turngray(image)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, threshold_1, threshold_2)
    return canny

def template_match(image, template, threshold=0.9):
    gray_image = turngray(image)
    gray_template = turngray(template)

    h, w = gray_template.shape[:2]
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)

    locations = np.where(result >= threshold)
    output = image.copy()

    for pt in zip(*locations[::-1]):
        cv2.rectangle(output, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return output


def resize(image, scale_factor=2, up_or_down="up"):
    output = image.copy()

    if up_or_down.lower() == "up":
        for _ in range(scale_factor):
            output = cv2.pyrUp(output)
    elif up_or_down.lower() == "down":
        for _ in range(scale_factor):
            output = cv2.pyrDown(output)

    return output

def main():
    lambo = cv2.imread("lambo.png")
    shapes = cv2.imread("shapes.png")
    shapestemplate = cv2.imread("shapes_template.png")

    sobel_img = sobel_edge_detection(lambo)
    saveimage(sobel_img, "lambosobel.png")

    canny_img = canny_edge_detection(lambo, 50, 50)
    saveimage(canny_img, "lambocanny.png")

    up_img = resize(lambo, scale_factor=1, up_or_down="up")
    down_img = resize(lambo, scale_factor=1, up_or_down="down")
    saveimage(up_img, "lamboup.png")
    saveimage(down_img, "lambodown.png")

    matched_img = template_match(shapes, shapestemplate)
    saveimage(matched_img, "shapesmatched.png")

if __name__ == "__main__":
    main()