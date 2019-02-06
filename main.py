import numpy
import cv2 as cv

from utils import Line, Color
from utils import Digit, closestDigit
from cnn import CNN


detectionDistance = 5
showing = True


print('=============INITIALIZING=============')
print('Showing Visuals:', showing)

nn = CNN('models/cnn.h5')

out = open('results/out.txt', 'w')
out.write('RA 195/2015 Bosko Vojinovic\n')
out.write('file\tsum\n')


for i in range(10):
    videoFile = 'data/video-' + str(i) + '.avi'
    print('=============', videoFile, '=============')
    cap = cv.VideoCapture(videoFile)
    _, image = cap.read()

    blueLine = Line(image, Color.BLUE)  # magic values AF
    greenLine = Line(image, Color.GREEN)
    print('Blue line: from ', blueLine.startPoint(), ' to ', blueLine.endPoint())
    print('Green line: from ', greenLine.startPoint(), ' to ', greenLine.endPoint())

    digits = []
    value = 0
    while True:

        status, image = cap.read()
        if (not status):
            break

        imageGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, imageBin = cv.threshold(imageGray, 180, 255, cv.THRESH_BINARY)
        imageBin = cv.dilate(imageBin, numpy.ones((3, 3)), iterations=2)
        #cv.imshow("2", imageBin)
        #cv.waitKey(1)
        contours, _ = cv.findContours(imageBin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        moved = []

        for contour in contours:
            bbox = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if area > 40:
                isFound, d = closestDigit(digits, bbox)
                if isFound:
                    d.bbox = bbox
                    moved.append(d)
                else:
                    x, y, w, h = bbox
                    region = imageGray[y:y+h+1, x:x+w+1]
                    digits.append(Digit(bbox, region))

        for d in digits:
            if(not moved.__contains__(d)):
                digits.remove(d)

        for d in digits:
            x, y, w, h = d.bbox
            if not d.blueCounted and blueLine.pointDistance((x + w, y + h)) < detectionDistance:
                region = nn.prepare(d.image)
                res = nn.predict(region)
                value += res[0]
                d.blueCounted = True
            if not d.greenCounted and greenLine.pointDistance((x + w, y + h)) < detectionDistance:
                region = nn.prepare(d.image)
                res = nn.predict(region)
                value -= res[0]
                d.greenCounted = True

        if (showing):
            display = image
            cv.putText(display, "SUM: " + str(value), (1, 25), cv.FONT_HERSHEY_PLAIN, 2, (255, 100, 100), 2)

            cv.line(display, blueLine.startPoint(), blueLine.endPoint(), (0, 0, 255), 2)
            cv.line(display, greenLine.startPoint(), greenLine.endPoint(), (0, 0, 255), 2)

            for d in digits:
                x, y, w, h = d.bbox
                cv.rectangle(display, (x, y), (x+w, y+h), (255, 100, 100), 1)

            wname = "Processing video: " + videoFile
            cv.imshow(wname, image)
            cv.moveWindow(wname, 100, 100)
            cv.waitKey(1)

    cv.destroyAllWindows()
    print('Final sum: ', value, '\n')
    out.write(videoFile + '\t' + str(value) + '\n')

out.close()
