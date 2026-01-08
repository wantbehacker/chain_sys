from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="ch",
    ocr_version="PP-OCRv4"
)

# C:\Users\Administrator\.paddlex\official_models\

# result = ocr.ocr("test.jpg", cls=True)
#
# for line in result[0]:
#     print(line[1][0])
