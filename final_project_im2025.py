#Tal Gorodetzky
import cv2
import numpy as np
import matplotlib.pyplot as plt


def gray_world_white_balance(image, alpha=0.7):
    """
    Applies Gray World White Balance with a correction factor.

    :param image: Input image in BGR format (as loaded by OpenCV).
    :param alpha: Correction factor (1.0 = full correction, 0.0 = no correction).
    :return: White-balanced image.
    """
    # Compute the mean values for each channel
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])

    # Compute the overall gray value
    avg_gray = (avg_r + avg_g + avg_b) / 3

    # Scaling factors with correction factor alpha
    scale_b = 1 + alpha * ((avg_gray / avg_b) - 1)
    scale_g = 1 + alpha * ((avg_gray / avg_g) - 1)
    scale_r = 1 + alpha * ((avg_gray / avg_r) - 1)

    # Apply scaling to each channel
    balanced_image = image.copy()
    balanced_image[:, :, 0] = np.clip(image[:, :, 0] * scale_b, 0, 255)
    balanced_image[:, :, 1] = np.clip(image[:, :, 1] * scale_g, 0, 255)
    balanced_image[:, :, 2] = np.clip(image[:, :, 2] * scale_r, 0, 255)

    return balanced_image.astype(np.uint8)

def enhance_contrast(image):
    """Enhance contrast using CLAHE in LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
def denoise_image(image):
    """Reduce noise using bilateral filtering."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def detect_colored_circles(image_path: str) -> list[dict]:
    """
    Detects colored circles on the membrane.

    Input:
    image_path (str): The path to an image file (JPEG or PNG) of the membrane.

    Output:
    A list of dictionaries, where each dictionary represents a detected circle
    with the keys: "x", "y", "radius", "color"
    """

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # clean the image
    image = gray_world_white_balance(image)
    image = denoise_image(image)
    image = enhance_contrast(image)
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges as separate lists
    circles = []
    color_bounds = [
        ("blue", np.array([100, 105, 60]), np.array([130, 255, 255])),
        ("light_red", np.array([0, 50, 50]), np.array([10, 255, 255])),
        ("dark_red", np.array([170, 50, 50]), np.array([180, 255, 255])),
        ("bright_yellow", np.array([25, 100, 70]), np.array([35, 255, 255])),
        ("dull_yellow", np.array([40, 32, 50]), np.array([90, 255, 255])),
        ("black", np.array([70, 40, 0]), np.array([120, 100, 95]))
    ]

    # Iterate over color bounds
    red_mask, yellow_mask = None, None  # Temporary masks for merging
    for color, lower, upper in color_bounds:
        mask = cv2.inRange(hsv, lower, upper)

        # Merge masks for red and yellow
        if color == "light_red":
            red_mask = mask
            continue
        elif color == "dark_red":
            mask = cv2.bitwise_or(red_mask, mask)
            color = "red"
        elif color == "bright_yellow":
            yellow_mask = mask
            continue
        elif color == "dull_yellow":
            mask = cv2.bitwise_or(yellow_mask, mask)
            color = "yellow"

        # Morphological operations to improve the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Detect contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours into circles
        for contour in contours:
            # Fit a circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            # Check for a reasonable radius size
            if radius < 35 or radius > 100:
                continue
            # Append to the list of circles
            circles.append({"x": int(x),"y": int(y),"radius": int(radius),"color": color})

    filtered = []
    for circle1 in circles:
        same_circle = False
        for circle2 in filtered:
            # Distance between centers
            distance = np.sqrt((circle1["x"] - circle2["x"]) ** 2 + (circle1["y"] - circle2["y"]) ** 2)
            if distance < circle1["radius"] or distance < circle2["radius"]:
                same_circle = True
                break
        if not same_circle:
            filtered.append(circle1)
    return filtered

def track_circles_over_time(image_paths: list[str]) -> dict:
    """
    Tracks circles over time across a sequence of images.

    Input:
    image_paths (list[str]): A list of image file paths representing the sequence of images over time.

    Output:
    A dictionary with two keys:
    - "table": A list of records (dictionaries) where each record includes:
        • image (int): The sequence number of the image.
        • circle_id (int): The identifier for the circle.
        • x (int): The x-coordinate of the circle’s center.
        • y (int): The y-coordinate of the circle’s center.
        • radius (int): The radius of the circle.
        • color (str): The color of the circle.
    """
    table = []
    circle_id_counter = 1
    previous_circles = []  # This will store circles from the previous image for comparison

    for image_index, image_path in enumerate(image_paths):
        # Detect circles in the current image
        circles = detect_colored_circles(image_path)

        # Match circles from the current image with previous circles
        for circle in circles:
            best_match = None
            min_distance = float('inf')

            # Compare with circles from the previous frame
            for previous_circle in previous_circles:
                distance = np.sqrt((circle["x"] - previous_circle["x"]) ** 2 + (circle["y"] - previous_circle["y"]) ** 2)
                if distance < min_distance and distance < previous_circle["radius"]:
                    best_match = previous_circle
                    min_distance = distance

            if best_match is not None:
                # If a match was found, update the circle with the same ID
                circle["circle_id"] = best_match["circle_id"]
            else:
                # If no match, assign a new circle ID
                circle["circle_id"] = circle_id_counter
                circle_id_counter += 1

            # Append the circle to the table
            table.append({"image": image_index + 1,"circle_id": circle["circle_id"],"x": circle["x"],"y": circle["y"],"radius": circle["radius"],"color": circle["color"]})
        # Update the previous_circles list for the next frame
        previous_circles = circles

    return {"table": table}

def show_results(image_path: str, circles: list) -> None:
    """
    Displays the detected circles on the original image using matplotlib.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Convert BGR to RGB for correct matplotlib display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Color mapping for visualization
    color_dict = {
        "red": "red",
        "yellow": "yellow",
        "blue": "blue",
        "black": "black",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(image_path)
    # Draw circles
    for i, circle in enumerate(circles):
        x, y, radius, color_name = circle["x"], circle["y"], circle["radius"], circle["color"]
        display_color = color_dict.get(color_name, "white")

        # Draw the circle
        circle_patch = plt.Circle((x, y), radius, color=display_color, fill=False, linewidth=2)
        ax.add_patch(circle_patch)

        # Draw center of the circle
        ax.scatter(x, y, color="yellow", s=10)

        # Add label
        label = f"{i + 1}: {color_name}"
        ax.text(x - 20, y - radius - 10, label, fontsize=10, color=display_color, backgroundcolor="white")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

#debug function
def get_hsv_range(image: np.ndarray, x: int, y: int, radius: int) -> tuple[list[int], list[int], list[int]]:
    """
    Given the x, y coordinates and radius of a circle, return the min, max, and average HSV values inside the circle.

    Input:
    - image (np.ndarray): The input image in BGR format.
    - x (int): X-coordinate of the circle's center.
    - y (int): Y-coordinate of the circle's center.
    - radius (int): Radius of the circle.

    Output:
    - (min_HSV, max_HSV, avg_HSV): A tuple containing:
      - min_HSV (list[int]): Minimum HSV values in the circle as [H, S, V].
      - max_HSV (list[int]): Maximum HSV values in the circle as [H, S, V].
      - avg_HSV (list[int]): Average HSV values in the circle as [H, S, V].
    """

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for the circle
    mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, thickness=-1)

    # Extract HSV values inside the circle
    hsv_values = hsv_image[mask == 255]

    if len(hsv_values) == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]  # Handle edge case where no pixels are inside the circle

    # Compute min, max, and average HSV values
    min_HSV = np.min(hsv_values, axis=0).tolist()
    max_HSV = np.max(hsv_values, axis=0).tolist()
    avg_HSV = np.mean(hsv_values, axis=0).astype(int).tolist()

    return min_HSV, max_HSV, avg_HSV

#same function, but with hough circles, work at most of the circles but not precise enough:
def classify_color_hsv(bgr):
    """
    Classifies a color based on its Hue (H), Saturation (S), and Value (V).

    Parameters:
        bgr (tuple): (B, G, R) values.

    Returns:
        str: Color name.
    """
    b, g, r = bgr

    # Convert BGR to HSV
    color_bgr = np.uint8([[[b, g, r]]])  # Shape (1,1,3)
    color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = color_hsv  # Extract Hue, Saturation, and Value

    # # Special condition for yellow
    if 30 < h <= 75 and 55 > s > 25 and 100 <= v <= 115:
        return "yellow"
    if 90<=h<=95 and 20<=s<=30 and 105<=v<=115:
        return "cyan"
    # If Saturation is too low, it's grayscale (white, gray, black)
    if s < 50:# and h<80:  # was 55
        if v < 50:
            return "black"
        elif v > 95:
            return "white"
        # elif h>=40:
        else:
            return "gray"
        # else:
        #     return "unknown"

    # Classify based on Hue ranges (OpenCV Hue range: 0-180)
    if 0 <= h < 15 or h >= 165:
        return "red"
    elif 15 <= h < 30 and (s<110 or 140<=s)  and 105<=v<=130:
        return "orange"
    elif 30 <= h < 45:
        return "yellow"
    elif 45 <= h < 90:
        return "green"
    elif 90 <= h < 130:
        return "cyan"
    elif 130 <= h < 165:
        return "blue"

    return "unknown"

def count_and_mark_circles(image_path, dp=2, min_dist=150, param1=10, param2=25, min_radius=30, max_radius=110, merge_threshold=100):
    """
    Counts the number of circles in an image using Hough Circle Transform, merging close circles, and marks them.

    Parameters:
        image_path (str): Path to the image file.
        dp (float): Inverse ratio of accumulator resolution to image resolution.
        min_dist (int): Minimum distance between detected circle centers.
        param1 (int): High threshold for Canny edge detection.
        param2 (int): Threshold for circle detection.
        min_radius (int): Minimum radius of detected circles.
        max_radius (int): Maximum radius of detected circles.
        merge_threshold (int): Max distance between two circles to be considered the same.

    Returns:
        int: Number of detected circles (after merging close ones).
        list[dict]: Details of detected circles.
    """
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_copy=image.copy()
    #Apply cleaning functions
    wb_image = gray_world_white_balance(image)
    contrast_image = enhance_contrast(wb_image)
    denoised_image = denoise_image(contrast_image)
    #image = sharpen_image(denoised_image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)  # Reduce noise
    #activate_orange_mode = 0
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integer values
        merged_circles = []

        for (x, y, radius) in sorted(circles[0], key=lambda c: (c[1], c[0])):  # Sort by y, then x
            found_close = False

            # Check if this circle is very close to an already detected one
            for mc in merged_circles:
                if abs(mc["x"] - x) < merge_threshold and abs(mc["y"] - y) < merge_threshold:
                    found_close = True
                    break

            if not found_close:
                # Create a mask for the circle
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), radius, 255, -1)

                # Get the average color inside the circle
                mean_color = cv2.mean(image, mask=mask)[:3]  # Get BGR color
                #start debug
                b, g, r = mean_color

                # Convert BGR to HSV
                color_bgr = np.uint8([[[b, g, r]]])  # Shape (1,1,3)
                color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0][0]

                h, s, v = color_hsv
                #end debug

                # Convert BGR to HSV and classify color
                color_name = classify_color_hsv(mean_color)

                if color_name == "white" or color_name == "unknown":
                    continue
                # Store circle details
                merged_circles.append({
                    "x": int(x),
                    "y": int(y),
                    "radius": int(radius),
                    "h": int(h),#debug
                    "s": int(s),#debug
                    "v": int(v),#debug
                    "color": color_name
                })

        #remove nearby circles
        updated_circles = merged_circles[:]  # Copy list for safe modification

        for circle1 in merged_circles[:]:  # Iterate safely
            for circle2 in merged_circles[:]:
                if circle1 == circle2:
                    continue  # Skip comparing the same circle

                # Check if circles are close
                if abs(circle1["x"] - circle2["x"]) <= 300 and abs(circle1["y"] - circle2["y"]) <= 300:

                    # Case 1: Both are yellow, keep the larger one
                    if circle1["color"] == "yellow" and circle2["color"] == "yellow":
                        if circle1["radius"] < circle2["radius"]:  # Remove smaller circle1
                            if circle1 in updated_circles:
                                updated_circles.remove(circle1)
                        elif circle1["radius"] > circle2["radius"]:  # Remove smaller circle2
                            if circle2 in updated_circles:
                                updated_circles.remove(circle2)
                        # If they have the same radius, keep one and remove the other
                        else:
                            if circle1["y"]<circle2["y"]:
                                if circle2 in updated_circles:
                                    updated_circles.remove(circle2)
                            else:
                                if circle1 in updated_circles:
                                    updated_circles.remove(circle1)


                    # Case 2: If one is yellow and the other is not, remove the yellow one
                    elif circle1["color"] == "yellow" and circle2["color"] != "yellow":
                        if circle1 in updated_circles:
                            updated_circles.remove(circle1)
                    elif circle2["color"] == "yellow" and circle1["color"] != "yellow":
                        if circle2 in updated_circles:
                            updated_circles.remove(circle2)

        merged_circles = updated_circles  # Update the list

        # Draw white circles around detected circles
        for circle in merged_circles:
            cv2.circle(image_copy, (circle["x"], circle["y"]), circle["radius"], (255, 255, 255), 2)  # White circle

        num_circles = len(merged_circles)  # Count merged circles
        detected_circles = merged_circles
    else:
        num_circles = 0

    # Convert BGR to RGB for correct color display in Matplotlib
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

    # Display the image with detected circles
    plt.imshow(image_rgb)
    plt.title(f"Detected Circles: {num_circles}")
    plt.show()

    return num_circles, detected_circles

if __name__ == "__main__":

    image_path = r"myImages\random_frames\random_002.jpg"
    image_paths = [rf"myImages\sequence_3\seq_{i:03}.jpg" for i in range(10)] #sequences between 1-4 (note that the membrane in seq2 does not contain circles)
    random_image_paths = [rf"myImages\random_frames\random_{i:03}.jpg" for i in range(10)] #for random sequence

    #part 1:
    circles = detect_colored_circles(image_path)
    for i in circles:
        print(i)
    show_results(image_path, circles)

    #part 2:
    results = track_circles_over_time(image_paths)
    for i in random_image_paths:
        circles = detect_colored_circles(i)
        show_results(i, circles)
    for i in results["table"]:
        print(i)


