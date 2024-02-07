import cv2
import numpy as np
import utilites as uts

filepath = "/Users/ahmadghanayem/Desktop/Technion/Project 2/utils/"

class ImageProcessor:
    def __init__(self):
        self.warped_frame = None
        self.roi_mask = None
        self.cleaned_mask_cc = None
        self.img_rgb = None
        self.prev_warped = None

    def get_warped(self):
        # Implementation of get_warped function
        return self.warped_frame

    def initial_detect(self):
        # Implementation of initial_detect function
        # This method should also store masks in self.masks
        self.load_masks()
        if self.roi_mask is None or self.cleaned_mask_cc is None:
            self.roi_mask, self.cleaned_mask_cc = get_masks(self.img_rgb)

    def detect_chessboard(self):
        # Implementation of detect_chessboard function
        try:
            self.warped_frame = warp_frame(self.img_rgb,self.roi_mask,self.cleaned_mask_cc)
        except cv2.error as e:
            self.wraped_frame = None
        return True if self.warped_frame is not None else False
        
    def save_masks(self):
        global filepath
        # Save the masks to a specified file
        np.save(filepath + '_roi_mask.npy', self.roi_mask)
        np.save(filepath + '_cleaned_mask_cc.npy', self.cleaned_mask_cc)

    def load_masks(self):
        global filepath
        # Loads the masks to a specified file
        try:
            self.roi_mask = np.load(filepath + '_roi_mask.npy')
            self.cleaned_mask_cc = np.load(filepath + '_cleaned_mask_cc.npy')
        except FileNotFoundError:
            self.roi_mask = None
            self.cleaned_mask_cc = None



# Function to extract squares from the board
def extract_squares(image,squares):
    squares_histograms = {}
    for square in squares:
        square_pixels = cropped_extract_shape_pixels(image,squares[square])
        squares_histograms[square] = square_pixels
    return squares_histograms

def cropped_extract_shape_pixels(image, shape_points):
    """
    Extracts the pixels inside a given shape from an image and crops the result to the shape's bounding box.

    :param image: Input image (2D or 3D).
    :param shape_points: List of boundary points of the shape [(x1, y1), (x2, y2), ...].
    :return: Cropped extracted region from the image inside the shape.
    """
    
    # Create a blank mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
   
    # Convert shape_points to an array and reshape it
    pts = np.array(shape_points).reshape((-1, 1, 2)).astype(int)
    pts = np.array([pts[0][0], pts[1][0], pts[3][0], pts[2][0]])
    pts = pts.reshape((-1, 1, 2))
    
    # Fill the shape in the mask with white
    cv2.fillPoly(mask, [pts], 255)
    
    # Extract the region from the original image using the mask
    extracted_region = cv2.bitwise_and(image, image, mask=mask)

    # Calculate bounding box of the mask to crop the region
    (y, x) = np.where(mask)
    (y1, y2, x1, x2) = (np.min(y), np.max(y), np.min(x), np.max(x))
    cropped_region = extracted_region[y1:y2, x1:x2]

    return cropped_region

def display_extracted_squares(squares):
    """
    Display each extracted square with its corresponding name.

    Parameters:
    - squares: Dictionary where keys are square names and values are the extracted square images.
    """
    
    for square_name, square_image in squares.items():
        cv2.imshow(f"Square: {square_name}: {square_image.shape}", square_image)
        cv2.waitKey(5000)  # Display for 1 second
        time.sleep(1)  # Additional delay to ensure visualization

    cv2.destroyAllWindows()



# Enhanced image processing
def enhance_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use CLAHE for adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur with a different kernel size to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Convert the grayscale image back to BGR
    bgr_img = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    
    return bgr_img



# Modified function from the code to detect chessboard lines
def get_masks(img):
    img_rgb = img.copy()

    # Convert to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Masking green and yellow regions
    lower_green = np.array([20, 50, 50])
    upper_green = np.array([90, 255, 255])
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([80, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    combined_mask = cv2.bitwise_or(mask_green, mask_yellow)



    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # Create ROI mask
    roi_mask = cv2.drawContours(np.zeros_like(combined_mask), [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Slightly dilate the mask
    kernel_small = np.ones((5,5), np.uint8)
    slightly_dilated_mask = cv2.dilate(combined_mask, kernel_small, iterations = 1)

    # Applying morphological opening to clean the small white dots
    kernel_opening = np.ones((5,5), np.uint8)
    cleaned_mask = cv2.morphologyEx(slightly_dilated_mask, cv2.MORPH_OPEN, kernel_opening)

    # Label the connected regions
    num_labels, labels = cv2.connectedComponents(cleaned_mask)

    # Initialize an empty mask for the cleaned output
    cleaned_mask_cc = np.zeros_like(cleaned_mask)

    # Define an area thresholdto filter small regions
    area_threshold = 115

    # Iterate over each label and filter based on area
    for label in range(1, num_labels):
        component_mask = (labels == label).astype(np.uint8) * 255
        component_area = np.sum(component_mask) / 255
        
        if component_area > area_threshold:
            cleaned_mask_cc = cv2.bitwise_or(cleaned_mask_cc, component_mask)

    return roi_mask,cleaned_mask_cc



def warp_frame(img_rgb,roi_mask,cleaned_mask_cc):

    # Extract the region
    extracted_region = cv2.bitwise_and(img_rgb, img_rgb, mask=roi_mask)

    # Subtract green and yellow regions to get the chessboard only
    better_chessboard_only = cv2.bitwise_and(extracted_region, extracted_region, mask=~cleaned_mask_cc)

    # Remove fully black pixels to get an image without the black background
    non_black_pixels_mask = np.any(better_chessboard_only != [0, 0, 0], axis=-1)
    rows = np.any(non_black_pixels_mask, axis=1)
    cols = np.any(non_black_pixels_mask, axis=0)
    no_black_background_img = better_chessboard_only[rows][:, cols]


    if np.size(no_black_background_img) == 0:
        return None

    # Convert to grayscale
    gray_no_black = cv2.cvtColor(no_black_background_img, cv2.COLOR_RGB2GRAY)


    # Find contours
    contours_no_black, _ = cv2.findContours(gray_no_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_no_black = max(contours_no_black, key=cv2.contourArea)

    # Approximate contour to polygon
    epsilon_no_black = 0.02 * cv2.arcLength(largest_contour_no_black, True)
    approx_corners_no_black = cv2.approxPolyDP(largest_contour_no_black, epsilon_no_black, True)
    
    # Reorder the vertices using the corrected custom criteria function
    custom_ordered_corners = dynamic_order_corners(np.array([point[0] for point in approx_corners_no_black]))

    # Perspective transformation using the custom ordered vertices
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    matrix_custom_ordered = cv2.getPerspectiveTransform(custom_ordered_corners, pts2)
    warped_chessboard_only = cv2.warpPerspective(no_black_background_img, matrix_custom_ordered, (500, 500))

    # Convert the image to grayscale
    gray_chessboard = cv2.cvtColor(warped_chessboard_only, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur
    blurred_chessboard = cv2.GaussianBlur(gray_chessboard, (3, 3), 0)

    # threshold the warped chessboard image
    thresh_ret, thresholded_warped_board = cv2.threshold(blurred_chessboard,110,255,cv2.THRESH_BINARY)
    #cv2.imshow('mask',cleaned_mask_cc)

    return warped_chessboard_only



def color_squares(square_num):
    for i in range(square_num):
        square_colors = square_colors + [(int(random.random()*255), int(random.random()*255), int(random.random()*255))]
    return square_colors

# Y-value-based corner ordering function
def y_based_order_corners(pts):
    if None in pts:
        return None
    # Convert the input format to a list of 2D points
    converted_pts = [tuple(point[0]) for point in pts]

    # Convert the input to a list of tuples for easy manipulation
    pts_list = [tuple(pt) for pt in converted_pts]
    
    # Sort points primarily by y-coordinate
    sorted_by_y = sorted(pts_list, key=lambda pt: pt[1])
    
    # Split into groups of 9 and sort each group by x-coordinate
    sorted_points = []
    for i in range(0, len(sorted_by_y), 9):
        group = sorted_by_y[i:i+9]
        sorted_group = sorted(group, key=lambda pt: pt[0])
        sorted_points.extend(sorted_group)
        
    return [(np.array(pt),) for pt in sorted_points]

# Dynamic corner ordering function
def dynamic_order_corners(pts):
    if len(pts) == 4:
        centroid = np.mean(pts, axis=0)
        sorted_pts = sorted(pts, key=lambda pt: (-np.arctan2(pt[1] - centroid[1], pt[0] - centroid[0])))
        return np.array([sorted_pts[i] for i in [3, 2, 0, 1]],dtype=np.float32)
    return None


def detect_circles(self, squares, img):
    """
    Detects circles in each square of the chessboard and determines their color.

    Parameters:
    - squares: Dictionary with square names as keys and their corners as values.
    - img: The image of the chessboard.

    Returns:
    - A dictionary with square names as keys. The values are lists of format [True/False, Color/None].
    """
    circle_results = {}

	# Determine the dominant color using the refined method

    for square_name, points in squares.items():
        # Extract the square region from the image
        x_start, y_start = int(points[0][0][0]), int(points[0][0][1])
        x_end, y_end = int(points[1][0][0]), int(points[2][0][1])
        
        square_region = img[y_start:y_end, x_start:x_end]
    
        if square_region.size == 0:
            print(f"Empty square_region for {square_name}!")
            continue

        # Convert the square region to grayscale
        gray_square = cv2.cvtColor(square_region, cv2.COLOR_RGB2GRAY)

        # Detect circles in the square
        circles = cv2.HoughCircles(gray_square, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=50, param2=30, minRadius=13, maxRadius=25)

        if circles is not None:
            # If a circle is detected, extract the circle region
            x, y, r = np.round(circles[0, 0]).astype("int")
            circle_region = square_region[y-r:y+r, x-r:x+r]
            
            # Draw the detected circle
            cv2.circle(img, (x_start + x, y_start + y), r, (0, 255, 0), 2)
            

            circle_results[square_name] = True
        else:
            circle_results[square_name] = False

    return circle_results


def detect_calib_circles(frame):
    """
    Detects up to two circles in the provided frame and returns their center coordinates.

    Parameters:
    - frame: The image frame in which to detect circles.

    Returns:
    - A list of (x, y) coordinates of the centers of up to two detected circles, otherwise None.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Hough Circle Detection
    circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=20, minRadius=5, maxRadius=16)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # To store the center coordinates of the circles
        centers = []

        # Iterate through detected circles (up to 2 circles)
        for i, (x, y, r) in enumerate(circles[:2]):
            # Draw the circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

            # Draw a label above the circle
            label = f"Radius: {r}"
            cv2.putText(frame, label, (x - 10, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Append the center to the list
            centers.append((x, y, r))

        return centers
    else:
        return None
        

def calculate_average_scale_factor(centers, physical_radius):
    """
    Calculates the average scale factor based on the detected radii and the known physical radius.

    Parameters:
    - centers: A list of tuples, each containing (x, y, r) of detected circles.
    - physical_radius: The known physical radius of the dots in millimeters.

    Returns:
    - The average scale factor in terms of mm/pixel.
    """
    if not centers or len(centers) == 0:
        return None

    radius = centers[0][2]
    scale_factor = physical_radius / radius  # Calculate scale factor for each circle
    return scale_factor
    
    def get_arm_pos(frame):
        circles = ip.detect_calib_circles(frame)
        if circles is None or len(circles) != 2:
            return None
        avg_sf = ip.calculate_average_scale_factor(circles,25)
        circles.sort(key=lambda x: x[1])
        constant_dot, movable_dot = circles[:2]
        constant_dot_converted = [constant_dot[1], constant_dot[0] * avg_sf]  # Swap and scale
        movable_dot_converted = [movable_dot[1], movable_dot[0] * avg_sf]     # Swap and scale

        # Relative position (movable - constant)
        relative_position = [movable_dot_converted[0] - constant_dot_converted[0],
        movable_dot_converted[1] - constant_dot_converted[1]]

        # Step 5: Calculate Arm's Movement
        # Assuming you have predefined values to reach the constant dot
        arm_movement_to_constant = [87, 0]  # Replace with your actual values
        arm_movement_to_movable = [arm_movement_to_constant[0] + relative_position[0],
        arm_movement_to_constant[1] + relative_position[1]]
        return arm_movement_to_movable
