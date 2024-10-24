import cv2 as cv
import numpy as np
from utility.utils import *
from utility.constants import *

def align_match_orb(image_to_align, reference_image, max_features=2000, min_match_quality=35, match_count_threshold=30):
    """
    Use ORB to align the image by finding good matches between keypoints of two images.
    """
    # Convert images to grayscale
    gray1 = cv.cvtColor(image_to_align, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(reference_image, cv.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv.ORB_create(nfeatures=max_features)

    # Detect keypoints and compute descriptors using ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if descriptors1 is None or descriptors2 is None:
        print("Error: Could not find descriptors in one of the images.")
        return None, None, None

    # Use BFMatcher with Hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Filter matches based on distance (match quality)
    good_matches = [m for m in matches if m.distance < min_match_quality]
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # If enough good matches, compute homography and align images
    if len(good_matches) > match_count_threshold:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Align the image
        height, width, _ = reference_image.shape
        aligned_image = cv.warpPerspective(image_to_align, M, (width, height))

        # Correct the order of the bounding box corners using perspective transformation
        pts = np.float32([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
        M_inv = np.linalg.inv(M)
        dst = cv.perspectiveTransform(pts, M_inv)

        # Calculate bounding box width (e.g., horizontal distance between top-left and top-right corners)
        bounding_box_width = np.linalg.norm(dst[0] - dst[1])

        #dst_corrected = cv.perspectiveTransform(pts, M_inv)

        # Draw bounding box around detected object in the camera frame
        matches_image = cv.polylines(image_to_align, [np.int32(dst)], True, (0, 255, 0), 3)

        return aligned_image, matches_image, bounding_box_width

    else:
        print(f"Not enough good matches: {len(good_matches)}/{match_count_threshold}")
        return None, None, None

def main():
    # Load the predefined trash image (reference image)
    reference_img = cv.imread('trash_image.png')

    # Initialize camera
    cap = cv.VideoCapture(0)

    if reference_img is None:
        print("Error: Could not load the trash image.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Try to align the current frame with the trash image
        result = align_match_orb(frame, reference_img)

        # Check if result is not None and contains the expected values
        if result is not None and len(result) == 3:
            aligned_image, matches_image, bounding_box_width = result

            if bounding_box_width is not None:
                print("Trash detected and aligned.")

                # Ensure the bounding_box_width reflects the size of the object in the frame
                if bounding_box_width != 0:
                    # Correct distance calculation
                    distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, bounding_box_width)

                    # Display the distance on the frame
                    cv.putText(matches_image, f"Distance: {distance:.2f} cm", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv.putText(matches_image, f"Bounding Box Width: {bounding_box_width:.2f}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Show the matches and the aligned image
                    cv.imshow('Aligned Image', aligned_image)
                    cv.imshow('Matches', matches_image)
            else:
                print("No good matches found.")
        else:
            print("Trash not detected or not enough good matches.")

        # Display the original camera frame
        cv.imshow('Camera Feed', frame)

        # Press 'q' to quit the loop
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
