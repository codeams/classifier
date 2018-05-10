import cv2


def display_image_with_label(image, label):
    cv2.putText(image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    cv2.imshow("Test-image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
