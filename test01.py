import cv2

img = cv2.VideoCapture(0)

while (True):
    s, frame = img.read()
    cv2.imshow("out",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('m'):
        break

img.release()
cv2.destroyAllWindows()