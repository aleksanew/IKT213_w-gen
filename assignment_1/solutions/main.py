import cv2

def print_image_information(image):
    height, width, channels = image.shape
    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
    print("Size:", image.size)
    print("Data type:", image.dtype)

def camera_informateion():
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    with open("camera_outputs.txt", "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"width: {int(width)}\n")
        f.write(f"height: {int(height)}\n")

    cap.release()

def main():
    img = cv2.imread("lena.png")
    print_image_information(img)
    camera_informateion()

if __name__ == "__main__":
    main()