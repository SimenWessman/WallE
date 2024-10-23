import cv2
import numpy as np

# Load the predefined trash image
trash_image = cv2.imread("trash_image.jpg", 0)  # Load as grayscale

# Initiate ORB detector for feature matching
orb = cv2.ORB_create()

# Find keypoints and descriptors for the trash image
kp_trash, des_trash = orb.detectAndCompute(trash_image, None)

# Create a Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Focal length and real-world width of the object (for distance calculation)
KNOWN_WIDTH = 5.0  # cm (adjust to your object size)
FOCAL_LENGTH = 615  # Assumed camera focal length in pixels (adjust as needed)


def calculate_distance(known_width, focal_length, per_width):
    # Distance calculation formula
    return (known_width * focal_length) / per_width


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the frame
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # Match descriptors between trash image and frame
    matches = bf.match(des_trash, des_frame)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches (for visualization)
    matched_image = cv2.drawMatches(trash_image, kp_trash, gray_frame, kp_frame, matches[:10], None, flags=2)

    # Extract the matched keypoints in the frame
    if len(matches) > 5:  # Arbitrary threshold for considering a valid match
        src_pts = np.float32([kp_trash[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find homography between the points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            h, w = trash_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # Draw the bounding box on the frame
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)

            # Approximate the width of the bounding box
            bounding_box_width = np.linalg.norm(dst[0] - dst[3])

            # Calculate distance
            distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, bounding_box_width)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with matches and distance
    cv2.imshow('Trash Detector', frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
