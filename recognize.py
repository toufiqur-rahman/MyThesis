import cv2
import os
import yaml

# video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training.yml')
#print(recognizer)

# faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Names corresponding to each id
names = []
for users in os.listdir("dataset"):
    names.append(users)
    print(users)

img = cv2.imread("test/chris.jpg")

while True:
    print("hello while loop")
    # _, img = video_capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray image", gray_img)
    cv2.waitKey(0)

    # faces = cv2.RectVector()
    # faces = faceCascade.detectMultiScale(gray_img, 1.2, 5)
    faces = faceCascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, _ = recognizer.predict(gray_img[y: y + h, x: x + w])
        if id:
            cv2.putText(img, names[id - 1], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Unknown", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),
                        1, cv2.LINE_AA)
    cv2.imshow("Recognize", img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

# cv2.waitKey(0)


# video_capture.release()
cv2.destroyAllWindows()
