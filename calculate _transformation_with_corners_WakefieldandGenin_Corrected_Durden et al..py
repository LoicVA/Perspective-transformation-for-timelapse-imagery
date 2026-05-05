#This script is a companion to Van Audenhaege et al. submitted.

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

############1. Parameters used to compute homography
vertical_acceptance=40 #31 #40 #Camera vertical acceptance angle
horizontal_acceptance=57 #42 #57 #Camera horizontal acceptance angle
camera_height=1450 #100 #1450  #Camera height above a flat seabed
camera_tilt_below_horizontal=45 #30 #45 #Camera tilt angle below horizontal
w=6000 #36 #6000 #Original image width
h=4000 #25 #4000 #Original image height
refractive_index=1.00 #1.00 for no refraction
path_of_files= "C:/Main/Articles/TMC timelapse/test/" #must contain the image and annotations to perspective transform
original_image="DSC_4407.JPG"
#Location where files will be saved

#############2. Compute the homography by calculating the corners
vertical_acceptance=(np.arcsin(math.sin(vertical_acceptance * math.pi / 180)/refractive_index))*180/math.pi
horizontal_acceptance=(np.arcsin(math.sin(horizontal_acceptance * math.pi / 180)/refractive_index))*180/math.pi

camera_tilt_below_horizontal_rad = (camera_tilt_below_horizontal * math.pi / 180)
vertical_acceptance_rad = (vertical_acceptance * math.pi / 180)
camera_tilt_below_horizontal_rad = (camera_tilt_below_horizontal * math.pi / 180)
horizontal_acceptance_rad = (horizontal_acceptance * math.pi / 180)

X = math.tan(vertical_acceptance_rad/2)*(h/(h/2)-1)
BU = (camera_height * (1 + X * math.tan(camera_tilt_below_horizontal_rad)) / (math.tan(camera_tilt_below_horizontal_rad) - X) - camera_height * (1 / math.tan(camera_tilt_below_horizontal_rad + vertical_acceptance_rad/ 2))) #Height of central meridian line
OB = camera_height/math.sin(camera_tilt_below_horizontal_rad+vertical_acceptance_rad/2)
JM = math.cos(vertical_acceptance_rad/2)*OB
Shb = (w)/(2*JM*math.tan(horizontal_acceptance_rad/2))
EF = w/Shb
OU = camera_height/math.sin(camera_tilt_below_horizontal_rad-vertical_acceptance_rad/2)
JL = math.cos(vertical_acceptance_rad/2)*OU
Shu = (w)/(2*JL*math.tan(horizontal_acceptance_rad/2))
ST = w/Shu

Area = (EF+ST)/2*BU

pts1 = np.float32([[0, 0], [w, 0],
                   [0, h], [w, h]])
pts2 = np.float32([[0, 0], [(ST), 0],
                   [(ST/2-EF/2), (BU)], [(ST/2+EF/2), (BU)]])

H=cv2.getPerspectiveTransform(src=pts1,dst=pts2)

#img=cv2.imread(path_of_files+original_image)
img=cv2.imread(path_of_files+original_image)
hh,ww=img.shape[:2]
width_pixels = int(np.ceil(ST))    # width in mm
height_pixels = int(np.ceil(BU))   # height in mm
img_perspective = cv2.warpPerspective(img, H, (width_pixels,height_pixels))
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img_perspective)
cv2.waitKey(0)
cv2.imwrite(path_of_files+"img_warped.jpg", img_perspective)

np.savetxt(path_of_files+"H_matrix.txt", H)

####Example of perspective transformation of a few points
##Calculate foreground area (where resolution < 0.25 mm2 per pixel)
pts1=np.array([[[0, 1570]]], dtype=np.float32)
p1=cv2.perspectiveTransform(pts1, H)
pts2=np.array([[[6000, 1570]]], dtype=np.float32)
p2=cv2.perspectiveTransform(pts2, H)
pts3=np.array([[[0, 4000]]], dtype=np.float32)
p3=cv2.perspectiveTransform(pts3, H)
pts4=np.array([[[6000, 4000]]], dtype=np.float32)
p4=cv2.perspectiveTransform(pts4, H)

polygon = np.array([p1, p2, p3, p4])
polygon = polygon.squeeze()
center = polygon.mean(axis=0)
angles = np.arctan2(polygon[:,1] - center[1],
                    polygon[:,0] - center[0])
polygon = polygon[np.argsort(angles)]

x = polygon[:, 0]
y = polygon[:, 1]

subset_area = 0.5 * abs(
    np.dot(x, np.roll(y, -1)) -
    np.dot(y, np.roll(x, -1))
)


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

mm_map, area_map = mm_per_pixel_map_from_homography_area(H, w, h, return_components=True)

mm_min, mm_max = np.min(area_map), np.max(area_map)
norm = Normalize(vmin=mm_min, vmax=mm_max)
colormap = plt.get_cmap('jet')

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(area_map, cmap=colormap, origin='upper')
ax.set_title("Perspective Resolution Heatmap (mm2/px)", fontsize=16)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Resolution (mm2/pixel)', rotation=270, labelpad=15)
plt.tight_layout()
plt.show()
fig.savefig(path_of_files + "perspective_heatmap_linear_scale.png", dpi=300, bbox_inches='tight')

fig2, ax2 = plt.subplots(figsize=(12, 8), constrained_layout=True)
im2 = ax2.imshow(mm_map, cmap=colormap, origin='upper', norm=norm)
ax2.set_title("Perspective Resolution Heatmap (mm2/px)", fontsize=16)
ax2.set_xlabel("X (pixels)")
ax2.set_ylabel("Y (pixels)")
mean_res_along_y = np.mean(area_map, axis=1)
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
ax_right.set_ylabel("Mean Resolution (mm2/pixel)", rotation=270, labelpad=15)
ax_right.tick_params(axis='y', colors='black', labelsize=9, length=5)
ax_right.spines['right'].set_visible(True)
ax_right.spines['right'].set_color('gray')
plt.show()
fig2.savefig(path_of_files + "perspective_heatmap_nonlinear_scale.png", dpi=300, bbox_inches='tight')

#######3. Overlay a grid
def overlay_grid(img_to_use, grid_spacing, color, thickness, save_path=None):
    image = img_to_use.copy()
    if image is None:
        print("Error: Unable to load image.")
        return

    height, width, _ = image.shape

    center_x = width // 2

    # --- Vertical lines (centered) ---
    x = center_x
    while x >= 0:
        cv2.line(image, (x, 0), (x, height), color, thickness)
        x -= grid_spacing

    x = center_x + grid_spacing
    while x <= width:
        cv2.line(image, (x, 0), (x, height), color, thickness)
        x += grid_spacing

    # --- Horizontal lines (bottom-aligned) ---
    y = height - 1
    while y >= 0:
        cv2.line(image, (0, y), (width, y), color, thickness)
        y -= grid_spacing

    cv2.imshow('Grid Overlay', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, image)

overlay_grid(img_perspective, grid_spacing=100, color=(0, 255, 0), thickness=2, save_path=path_of_files+"gridded_img.jpg")

####4. Draw the grid on the original image

class PerspectiveGrid:
    def __init__(self, grid_spacing=100, color=(0, 255, 0), thickness=2):
        self.grid_spacing = grid_spacing
        self.color = color
        self.thickness = thickness

    def draw_on_image(self, image):
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        center_x = w // 2

        # --- Vertical lines (centered) ---
        x = center_x
        while x >= 0:
            cv2.line(img_copy, (x, h - 1), (x, 0), self.color, self.thickness)
            x -= self.grid_spacing

        x = center_x + self.grid_spacing
        while x < w:
            cv2.line(img_copy, (x, h - 1), (x, 0), self.color, self.thickness)
            x += self.grid_spacing

        # --- Horizontal lines (bottom-aligned) ---
        y = h - 1
        while y >= 0:
            cv2.line(img_copy, (0, y), (w, y), self.color, self.thickness)
            y -= self.grid_spacing

        return img_copy

    def project_to_original(self, perspective_shape, H_inv, original_img):
        h_persp, w_persp = perspective_shape[:2]
        img_warped = original_img.copy()

        center_x = w_persp // 2

        # --- Vertical lines (centered) ---
        x = center_x
        while x >= 0:
            pts = np.float32([[x, h_persp - 1], [x, 0]]).reshape(-1, 1, 2)
            warped_line = cv2.perspectiveTransform(pts, H_inv)
            pt1, pt2 = map(tuple, np.int32(warped_line[:, 0]))
            cv2.line(img_warped, pt1, pt2, self.color, self.thickness)
            x -= self.grid_spacing

        x = center_x + self.grid_spacing
        while x < w_persp:
            pts = np.float32([[x, h_persp - 1], [x, 0]]).reshape(-1, 1, 2)
            warped_line = cv2.perspectiveTransform(pts, H_inv)
            pt1, pt2 = map(tuple, np.int32(warped_line[:, 0]))
            cv2.line(img_warped, pt1, pt2, self.color, self.thickness)
            x += self.grid_spacing

        # --- Horizontal lines (bottom-aligned) ---
        y = h_persp - 1
        while y >= 0:
            pts = np.float32([[0, y], [w_persp - 1, y]]).reshape(-1, 1, 2)
            warped_line = cv2.perspectiveTransform(pts, H_inv)
            pt1, pt2 = map(tuple, np.int32(warped_line[:, 0]))
            cv2.line(img_warped, pt1, pt2, self.color, self.thickness)
            y -= self.grid_spacing

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

# Project the gridded image back to original dimensions
def reverse_homography(img, homography_matrix, img_original):
    # Step 1: Invert the homography matrix
    inv_homography = np.linalg.inv(homography_matrix)

    # Step 2: Apply the inverse homography to the image
    height, width = img_original.shape[:2]
    restored_image = cv2.warpPerspective(img, inv_homography, (width, height))

    return restored_image

# Example usage
# Assume homography_matrix is the matrix you already have
restored_img = reverse_homography(gridded_perspective, H, img_original=img)

# Display the restored image
cv2.imshow('Restored Image', restored_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite(path_of_files+"gridded_img_inverted.jpg", restored_img)

#############5. Convert annotations
import numpy as np
import pandas as pd

# Load the CSV with correct delimiter
df = pd.read_csv(path_of_files+"Total_untransformed.csv", delimiter=',')

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
df.to_csv(path_of_files+"Total_transformed.csv", index=False)