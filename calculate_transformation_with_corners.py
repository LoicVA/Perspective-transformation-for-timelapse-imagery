#This script is a companion to Van Audenhaege et al. in prep.
#Please, use the following reference when making use of this script:
#Reference to add###

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

############1. Parameters used to compute homography
vertical_acceptance=40 #Camera vertical acceptance angle
horizontal_acceptance=57 #Camera horizontal acceptance angle
camera_height=1450  #Camera height above a flat seabed
camera_tilt_below_horizontal=45 #Camera tilt angle below horizontal
w=6000 #Original image width
l=4000 #Original image height
refractive_index=1.00 #1.00 for no refraction
path_of_files= "FOLDER" #must contain the image and annotations to perspective transform
original_image="DSC_4407.JPG"
#Location where files will be saved

#############2. Compute the homography
vertical_acceptance=(np.arcsin(math.sin(vertical_acceptance * math.pi / 180)/refractive_index))*180/math.pi
horizontal_acceptance=(np.arcsin(math.sin(horizontal_acceptance * math.pi / 180)/refractive_index))*180/math.pi
beta = 90-camera_tilt_below_horizontal

A=math.tan((beta+vertical_acceptance/2) * math.pi / 180)*camera_height
B=math.tan((beta-vertical_acceptance/2) * math.pi / 180)*camera_height
C=camera_height/math.cos((beta+vertical_acceptance/2) * math.pi / 180)
D=camera_height/math.cos((beta-vertical_acceptance/2) * math.pi / 180)
E=math.tan((horizontal_acceptance/2) * math.pi / 180)*C
F=math.tan((horizontal_acceptance/2) * math.pi / 180)*D

pts1 = np.float32([[0, 0], [w, 0],
                   [0, l], [w, l]])
pts2 = np.float32([[0, 0], [(E+E), 0],
                   [(E-F), (A-B)], [(E+F), (A-B)]])

H=cv2.getPerspectiveTransform(src=pts1,dst=pts2)

img=cv2.imread(path_of_files+original_image)
hh,ww=img.shape[:2]
width_pixels = int(np.ceil(2*E))    # E in mm
height_pixels = int(np.ceil(A-B))   # height in mm
img_perspective = cv2.warpPerspective(img, H, (width_pixels,height_pixels))
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img_perspective)
cv2.waitKey(0)
cv2.imwrite(path_of_files+"img_warped.jpg", img_perspective)

Area = ((E+E)+(F+F))/2*(A-B)
np.savetxt(path_of_files+"H_matrix.txt", H)

####2. With grid with each cell being a pixel, we calculate the areal distortion to map the resolution
def mm_per_pixel_map_from_homography_area(H, img_width, img_height, return_components=False):
    # Create grid of pixel corner coordinates
    u = np.arange(img_width + 1)
    v = np.arange(img_height + 1)
    Uc, Vc = np.meshgrid(u, v)
    corners = np.stack([Uc, Vc], axis=-1).reshape(-1, 2).astype(np.float64)

    # Transform all corner points with homography
    corners_h = cv2.perspectiveTransform(corners[None, :, :], H)[0]

    # Reshape back to grid
    X = corners_h[:, 0].reshape(img_height + 1, img_width + 1)
    Y = corners_h[:, 1].reshape(img_height + 1, img_width + 1)

    # Compute area of each pixel’s transformed quadrilateral
    # pixel (i, j) -> corners: (i, j), (i+1, j), (i+1, j+1), (i, j+1)
    x0, y0 = X[:-1, :-1], Y[:-1, :-1]
    x1, y1 = X[:-1, 1:],  Y[:-1, 1:]
    x2, y2 = X[1:, 1:],   Y[1:, 1:]
    x3, y3 = X[1:, :-1],  Y[1:, :-1]

    # Shoelace formula for quadrilateral area
    area = 0.5 * np.abs(
        (x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0) -
        (y0 * x1 + y1 * x2 + y2 * x3 + y3 * x0)
    )

    # Convert area (mm² per pixel²) → scalar mm/px equivalent
    mm_per_px_scalar = np.sqrt(area)

    if return_components:
        return mm_per_px_scalar, area
    else:
        return mm_per_px_scalar

mm_map, area_map = mm_per_pixel_map_from_homography_area(H, w, l, return_components=True)

mm_min, mm_max = np.min(mm_map), np.max(mm_map)
norm = Normalize(vmin=mm_min, vmax=mm_max)
colormap = plt.get_cmap('jet')

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(mm_map, cmap=colormap, origin='upper')
ax.set_title("Perspective Resolution Heatmap (mm/px)", fontsize=16)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Resolution (mm/pixel)', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()
fig.savefig(path_of_files + "perspective_heatmap_linear_scale.png", dpi=300, bbox_inches='tight')

fig2, ax2 = plt.subplots(figsize=(12, 8), constrained_layout=True)
im2 = ax2.imshow(mm_map, cmap=colormap, origin='upper', norm=norm)
ax2.set_title("Perspective Resolution Heatmap (mm/px)", fontsize=16)
ax2.set_xlabel("X (pixels)")
ax2.set_ylabel("Y (pixels)")
mean_res_along_y = np.mean(mm_map, axis=1)
ax_right = ax2.twinx()
ax_right.set_ylim(ax2.get_ylim())
min_val = np.min(mean_res_along_y)
max_val = np.max(mean_res_along_y)
tick_min = np.floor(min_val * 100) / 100.0
tick_max = np.ceil(max_val * 100) / 100.0
tick_vals = np.arange(tick_min, tick_max + 1e-9, 0.01)
y_positions = [np.argmin(np.abs(mean_res_along_y - t)) for t in tick_vals]
ax_right.set_yticks(y_positions)
ax_right.set_yticklabels([f"{t:.2f}" for t in tick_vals])
ax_right.set_ylabel("Mean Resolution (mm/pixel)", rotation=270, labelpad=15)
ax_right.tick_params(axis='y', colors='black', labelsize=9, length=5)
ax_right.spines['right'].set_visible(True)
ax_right.spines['right'].set_color('gray')
plt.show()
fig2.savefig(path_of_files + "perspective_heatmap_nonlinear_scale.png", dpi=300, bbox_inches='tight')

#######3. Overlay a grid
def overlay_grid(img_to_use, grid_spacing, color, thickness, save_path):
    image = img_to_use.copy()
    if image is None:
        print("Error: Unable to load image.")
        return

    height, width, _ = image.shape

    for x in range(width - grid_spacing, 0, -grid_spacing):
        cv2.line(image, (x, 0), (x, height), color, thickness)

    for y in range(height - grid_spacing, 0, -grid_spacing):
        cv2.line(image, (0, y), (width, y), color, thickness)

    cv2.imshow('Grid Overlay', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")

overlay_grid(img_perspective, grid_spacing=100, color=(0, 255, 0), thickness=2, save_path=path_of_files+"gridded_img.jpg")

####4. Draw the grid on the original image

class PerspectiveGrid:
    def __init__(self, grid_spacing=100, color=(0, 255, 0), thickness=2):
        self.grid_spacing = grid_spacing
        self.color = color
        self.thickness = thickness

    def draw_on_image(self, image):
        """
        Draw a grid starting from the bottom-left corner,
        incrementing upwards and to the right.
        """
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        # Vertical lines: left → right
        for x in range(0, w, self.grid_spacing):
            cv2.line(img_copy, (x, h - 1), (x, 0), self.color, self.thickness)

        # Horizontal lines: bottom → top
        for y in range(h - 1, -1, -self.grid_spacing):
            cv2.line(img_copy, (0, y), (w, y), self.color, self.thickness)

        return img_copy

    def project_to_original(self, perspective_shape, H_inv, original_img):
        """
        Project the grid from the perspective image back to the original image
        using the inverse homography.
        """
        h_persp, w_persp = perspective_shape[:2]
        img_warped = original_img.copy()

        # Generate vertical lines in perspective image
        for x in range(0, w_persp, self.grid_spacing):
            pts = np.float32([[x, h_persp - 1], [x, 0]]).reshape(-1, 1, 2)
            warped_line = cv2.perspectiveTransform(pts, H_inv)
            pt1 = tuple(np.int32(warped_line[0, 0]))
            pt2 = tuple(np.int32(warped_line[1, 0]))
            cv2.line(img_warped, pt1, pt2, self.color, self.thickness)

        # Generate horizontal lines in perspective image
        for y in range(h_persp - 1, -1, -self.grid_spacing):
            pts = np.float32([[0, y], [w_persp - 1, y]]).reshape(-1, 1, 2)
            warped_line = cv2.perspectiveTransform(pts, H_inv)
            pt1 = tuple(np.int32(warped_line[0, 0]))
            pt2 = tuple(np.int32(warped_line[1, 0]))
            cv2.line(img_warped, pt1, pt2, self.color, self.thickness)

        return img_warped

grid = PerspectiveGrid(grid_spacing=100, color=(0, 255, 0), thickness=2)
gridded_perspective = grid.draw_on_image(img_perspective)
cv2.imshow("Grid on Perspective", gridded_perspective)

# Compute inverse homography
H_inv = np.linalg.inv(H)

# Project grid back to original
gridded_original = grid.project_to_original(img_perspective.shape, H_inv, img)
cv2.imshow("Grid Back on Original", gridded_original)
cv2.imwrite(path_of_files+"gridded_img_original.jpg", gridded_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

#############5. Convert annotations
import numpy as np
import pandas as pd

# Load the CSV with correct delimiter
#df = pd.read_csv("C:\Main\Articles\TMC timelapse\Annex C Supplementary data/v2/Total_untransformed.csv", delimiter=',')
#df = pd.read_csv("C:\Main\Articles\TMC timelapse\Width_of_clearance/Results.csv", delimiter=',')
df = pd.read_csv(path_of_files+"area_considered_for_annotations.csv", delimiter=',')

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
df.to_csv(path_of_files+"transformed_area_considered_for_annotations.csv", index=False)