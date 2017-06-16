import cv2
from keras.models import load_model
import numpy as np


cap = cv2.VideoCapture(0)
model = load_model('tl_models/tl28')
dic = {0: 'Angry',
       1: 'Disgust',
       2: 'Fear',
       3: 'Happy',
       4: 'Sad',
       5: 'Surprise',
       6: 'Neutral'}


def face_detect(image, cascPath):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    return faces


def preprocess(image):
    image = np.array(cv2.resize(image, (48, 48)), dtype='float64')
    mean = np.mean(image, keepdims=True)
    image -= mean
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=-1)
    return image


def facial_expression(image):
    pred = model.predict(image)
    return pred


n = 0
texts = []
scale = 0.15
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect(gray, 'haarcascade_frontalface_default.xml')
    
    if len(faces):
        x, y, w, h = faces[-1]
        x -= int(scale * w)
        y -= int(scale * h)
        w, h = int((1 + 2 * scale) * w), int((1 + 2 * scale) * h)

        # print faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        face = gray[x:x + w, y:y + h]
        if n % 5 == 0 and face.shape[0] > 0 and face.shape[1] > 0:
            face = preprocess(face)
            pred = facial_expression(face)[0]
            output = sorted([(dic[i], pred[i]) for i in range(len(pred))], key=lambda m: m[1], reverse=True)
            texts = [" ".join([a, "%.3f" % b]) for a, b in output]
            # print(texts)
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (x+w, y+i*40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    n += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
