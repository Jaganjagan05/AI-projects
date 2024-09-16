import cv2
import numpy as np

# Load class labels from MobileNet-SSD
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"]

# Load the MobileNet SSD model from the disk
model = r"C:\Users\majag\Downloads\mobilenet_iter_73000.caffemodel"
config = r"C:\Users\majag\Downloads\deploy.prototxt"

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(config, model)

# Open a video capture (you can also use a video file instead of 0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read the frame from the video feed
    ret, frame = cap.read()
    if not ret:
        break

    # Get the height and width of the frame
    h, w = frame.shape[:2]

    # Prepare the image for MobileNet SSD
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # Set the input to the pre-trained deep learning network
    net.setInput(blob)

    # Forward pass: get the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence of the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections
        if confidence > 0.2:  # You can adjust the threshold based on requirements
            # Get the class label index and bounding box coordinates
            class_index = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label on the frame
            label = f"{class_labels[class_index]}: {confidence * 100:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
