import cv2
import numpy as np
import math
img_path="C:/Main/Articles/TMC timelapse/DSC_4407.jpg"

vertical_acceptance=40
horizontal_acceptance=57
camera_height=1450
camera_tilt_below_horizontal=45
w=6000 #Original image width
l=4000 #Original image height
L=1000 #orizontal offset for plotting
refractive_index=1.00 #None since already corrected angles

vertical_acceptance=(np.arcsin(math.sin(vertical_acceptance * math.pi / 180)/refractive_index))*180/math.pi
horizontal_acceptance=(np.arcsin(math.sin(horizontal_acceptance * math.pi / 180)/refractive_index))*180/math.pi

A=math.tan((90-camera_tilt_below_horizontal+vertical_acceptance/2) * math.pi / 180)*camera_height
B=math.tan((90-camera_tilt_below_horizontal) * math.pi / 180)*camera_height
C=math.tan((90-camera_tilt_below_horizontal-vertical_acceptance/2) * math.pi / 180)*camera_height
D=camera_height/math.cos((90-camera_tilt_below_horizontal) * math.pi / 180)
E=math.tan((horizontal_acceptance/2) * math.pi / 180)*D

offset=L
pts1 = np.float32([[0, l/2], [w/2, l],
                   [w, l/2], [w/2, 0]])
pts2 = np.float32([[0+offset, -(B-C)+(A-C)], [E+offset, A-C],
                   [E*2+offset, -(B-C)+(A-C)], [E+offset, 0]])
#+(A-C) is to flip vertically
H=cv2.getPerspectiveTransform(src=pts1,dst=pts2)

img=cv2.imread(img_path)
#cv2.imshow("result", img)
#cv2.waitKey(0)
hh,ww=img.shape[:2]
img_perspective = cv2.warpPerspective(img, H, (int(E*2+2*offset),int(A-C)))
cv2.imshow("result", img_perspective)
cv2.waitKey(0)
cv2.imwrite("C:/Main/Articles/TMC timelapse/img_warped.jpg", img_perspective)

#######Overlay a grid
def overlay_grid(img_to_use, grid_spacing, color, thickness, save_path):
    # Load the image
    image = img_to_use.copy()
    if image is None:
        print("Error: Unable to load image.")
        return

    height, width, _ = image.shape

    # Draw vertical lines (starting from right to left)
    for x in range(width - grid_spacing, 0, -grid_spacing):
        cv2.line(image, (x, 0), (x, height), color, thickness)

    # Draw horizontal lines (starting from bottom to top)
    for y in range(height - grid_spacing, 0, -grid_spacing):
        cv2.line(image, (0, y), (width, y), color, thickness)

    # Display the image with the grid overlay
    cv2.imshow('Grid Overlay', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image if a save path is provided
    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")

overlay_grid(img_perspective, grid_spacing=100, color=(0, 255, 0), thickness=2, save_path="C:/Main/Articles/TMC timelapse/gridded_img.jpg")

############Transformed back the gridded image
def reverse_homography(img, homography_matrix, img_original):
    # Step 1: Invert the homography matrix
    inv_homography = np.linalg.inv(homography_matrix)

    # Step 2: Apply the inverse homography to the image
    height, width = img_original.shape[:2]
    restored_image = cv2.warpPerspective(img, inv_homography, (width, height))

    return restored_image


# Example usage
gridded_img = cv2.imread("C:/Main/Articles/TMC timelapse/gridded_img.jpg")  # Load your image
# Assume homography_matrix is the matrix you already have
restored_img = reverse_homography(gridded_img, H, img_original=img)

# Display the restored image
cv2.imshow('Restored Image', restored_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("C:/Main/Articles/TMC timelapse/gridded_img_inverted.jpg", restored_img)

np.savetxt("C:/Main/Articles/TMC timelapse/H_matrix.txt", H)

########################Convert annotations
import numpy as np
import pandas as pd

# Load the CSV with correct delimiter
df = pd.read_csv("C:\Main\Articles\TMC timelapse\Annex C Supplementary data/v1/Total_untransformed.csv", delimiter=',')

# Prepare the coordinates as homogeneous coordinates (x, y, 1)
coords = np.vstack((df['X'], df['Y'], np.ones(len(df))))

# Apply the homography
transformed_coords = H @ coords

# Normalize to convert back from homogeneous coordinates
transformed_coords /= transformed_coords[2, :]

# Add transformed coordinates to the DataFrame
df['X_transformed'] = transformed_coords[0]
df['Y_transformed'] = transformed_coords[1]

# Save to a new CSV
df.to_csv("C:\Main\Articles\TMC timelapse\Annex C Supplementary data/v1/Total_transformed.csv", index=False)