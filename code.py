import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)  # Query image
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)  # Train image

# Initialize ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create a BFMatcher object with distance measurement cv2.NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw top 10 matches
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
plt.figure(figsize=(10, 5))
plt.imshow(img_matches)
plt.title('Feature Matching using ORB')
plt.axis('off')
plt.show()