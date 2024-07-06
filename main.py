import cv2
import mediapipe as mp

# Video input stream from webcam
vis = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingerStatus = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbStatus = (4, 2)


while True:
    # Read in image
    success, img = vis.read()

    # Convert image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process Image
    results = hands.process(imgRGB)
    multiLandMarks = results.multi_hand_landmarks
    jointLocations = []
    totalCount = 0

    if multiLandMarks:
        # [{1st hand coords}, {2nd hand coords}]
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            # Process joints
            for idx, lm in enumerate(handLms.landmark):
                height, width, colorChannel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                jointLocations.append((cx, cy))
        for point in jointLocations:
            # Draw circle on joint
            cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

        # Count fingers
        for coords in fingerStatus:
            if jointLocations[coords[0]][1] < jointLocations[coords[1]][1]: # top joint y coord < bottom joint y coord
                totalCount += 1

        if jointLocations[thumbStatus[1]][0] < jointLocations[thumbStatus[0]][0]:
            totalCount += 1




    # Display image
    cv2.putText(img, str(totalCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)
    cv2.imshow("Finger Counter", img)
    cv2.waitKey(5)