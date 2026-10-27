import pytesseract
import cv2

class PersianOCR:
    def __init__(self):
        pass

    def extract_text(self, image):
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OCR (Persian + English together)
        text = pytesseract.image_to_string(gray, lang="fas+eng")

        return text.strip()