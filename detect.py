import cv2
import sys
from os import listdir
from os.path import isfile, join, dirname

count = 0
dataset_origin = sys.argv[1]
dest = join(dataset_origin, "faces")
cascade_file = join(dirname(__file__), "lbpcascade_animeface.xml")
cascade = cv2.CascadeClassifier(cascade_file)

def detect(filename):
    global count
    print(filename)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (24, 24))
    for (x, y, w, h) in faces:
        y = y - 80
        x = x - 20
        w = w + 80
        h = h + 80
        face_crop = image[y:y+w, x:x+h]
        cv2.imwrite(dest + "/" + str(count) + ".jpg", face_crop)
        count = count + 1

    cv2.waitKey(0)

files = [join(dataset_origin, f) for f in listdir(dataset_origin) if isfile(join(dataset_origin, f))]
for f in files:
    detect(f)