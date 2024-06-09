import tkinter as tk
import cv2
import threading
import numpy as np
from sklearn.cluster import KMeans
import math
from py_openshowvar import openshowvar

# Flag to control the video loop
video_running = True

# Robot communication setup
robot = openshowvar('192.168.1.1', 7000)  # Replace with your robot's IP and port

try:
    robot.can_connect()
    print("Connection to the robot established successfully.")
except Exception as e:
    print(f"Failed to connect to the robot: {e}")

# Global variable for selected draw type
selected_draw_type = 1
origin_found = False
origin_x = 0
origin_y = 0

###### SEARCH CAMERA #########

# Function to find the USB camera index (assuming it is not 0)
def find_usb_camera_index():
    for index in range(1, 10):  # Start from 1 to skip the built-in camera
        vid = cv2.VideoCapture(index)
        if vid.isOpened():
            vid.release()
            return index
    return -1

######## ORIENTATION OF ROBOT ###################

def determine_orientation(circles):
    if len(circles) < 2:
        return "Undefined"

    # Calculate the angle between the first two circles and the horizontal axis
    x1, y1 = circles[0][:2]
    x2, y2 = circles[1][:2]

    try:
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    except ZeroDivisionError:
        # Handle the case where the denominator is zero
        return "Undefined"
    except OverflowError:
        # Handle overflow error
        return "Overflow"

    # Normalize the angle to be within -180 to 180 degrees
    angle = (angle + 180) % 360 - 180

    # Determine orientation based on the angle with buffer zones
    if -15 <= angle <= 15 or 165 <= angle <= 180 or -180 <= angle <= -165:
        return "H"  # Horizontal
    elif 75 <= angle <= 105 or -105 <= angle <= -75:
        return "V"  # Vertical
    elif 15 < angle < 75:
        return "DR"  # Diagonal right
    elif -75 < angle < -15:
        return "DL"  # Diagonal left
    else:
        return "Unknown"

#### VIDEO CAPTURE AND PROCESSING ####

# Function to capture video and detect circles and rectangles
def video_capture():
    global video_running, origin_found, origin_x, origin_y
    camera_index = find_usb_camera_index()
    if camera_index == -1:
        print("No camera found")
        return

    vid = cv2.VideoCapture(camera_index)
    if not vid.isOpened():
        print("Could not open camera")
        return

    while video_running:
        ret, frame = vid.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        # Detect the black square first
        if not origin_found:
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Assuming square has 4 corners
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    if 0.9 <= aspect_ratio <= 1.1:  # Ensure the aspect ratio is close to 1
                        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                        origin_x = x + w // 2
                        origin_y = y + h // 2
                        cv2.circle(frame, (origin_x, origin_y), 5, (255, 255, 0), -1)
                        origin_found = True
                        print(f"Black square detected at ({origin_x}, {origin_y})")
                        break

        if origin_found:
            # Detect circles in the frame
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=30,
                param1=50, 
                param2=30, 
                minRadius=15, 
                maxRadius=30
            )

            lego_count = 0

            if circles is not None:  # If circles are detected
                circles = np.uint16(np.around(circles))

                if len(circles[0]) >= 4:  # Ensure there are enough circles to form Legos
                    n_clusters = len(circles[0]) // 4
                    if n_clusters > 0:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        kmeans.fit(circles[0][:, :2])  # Fit only x, y coordinates
                        labels = kmeans.labels_

                        # Draw each group as a separate Lego
                        for i in range(max(labels) + 1):
                            lego_circles = circles[0][labels == i]

                            if len(lego_circles) > 1:
                                lego_count += 1
                                sorted_lego_circles = sorted(lego_circles, key=lambda x: (x[0], x[1]))  # Sort by x, then y

                                for (x, y, r) in sorted_lego_circles:
                                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Draw circle
                                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Draw center

                                # Calculate the midpoint of the square
                                mid_x = int(sum([coord[0] for coord in sorted_lego_circles]) / 4)
                                mid_y = int(sum([coord[1] for coord in sorted_lego_circles]) / 4)
                                cv2.circle(frame, (mid_x, mid_y), 5, (255, 255, 0), -1)  # Draw midpoint

                                # Calculate coordinates relative to the origin
                                rel_x = mid_x - origin_x
                                rel_y = mid_y - origin_y

                                # Determine the orientation
                                orientation = determine_orientation(sorted_lego_circles)
                                print(f"Lego {lego_count} at relative position ({rel_x}, {rel_y}) with orientation: {orientation}")

                                # Extract color information and print the color of the lego
                                lego_color = extract_lego_color(frame)

                                # Send the coordinates and additional data to the robot
                                send_robot_approx(rel_x, rel_y, lego_color, orientation)

            if lego_count > 0:
                print(f"Total Legos detected: {lego_count}")

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            break

    vid.release()
    cv2.destroyAllWindows()


# Function to extract the color of the lego
def extract_lego_color(frame):
    # Setting values for base colors
    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]

    # Computing the mean
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    # Displaying the most prominent color
    if (b_mean > g_mean and b_mean > r_mean):
        color = "Blue"
    elif (g_mean > r_mean and g_mean > b_mean):
        color = "Green"
    else:
        color = "Red"
    
    send_color(color)  # Send color to the robot
    return color

######## Robot Communication ##########

# Send midpoint coordinates to the robot
def send_midpoint(mid_x, mid_y):
    try:
        # Convert the coordinates to a string
        coordinates = f"{mid_x},{mid_y}"
        
        # Send the coordinates to the robot
        robot.write("COORDINATES", coordinates)
        
    except Exception as e:
        print("Error:", e)

# Send orientation to the robot
def send_orientation(orientation):
    try:
        robot.write("ORIENTATION", orientation)
    except Exception as e:
        print("Error:", e)

# Send color to the robot
def send_color(color):
    try:
        robot.write("COLOR", color)
    except Exception as e:
        print("Error:", e)

# Send approximate lego position, color, and orientation to the robot
def send_robot_approx(mid_x, mid_y, color, orientation):
    try:
        lego_height = 4
        approach_height_above_lego = 1
        approach_z = -(lego_height + approach_height_above_lego)
        goal_z = 0
        
        if selected_draw_type == 1:
            color_order = ["Red", "Blue", "Green"]
        elif selected_draw_type == 2:
            color_order = ["Blue", "Green", "Red"]
        elif selected_draw_type == 3:
            color_order = ["Green", "Red", "Blue"]

        if color in color_order:
            priority = color_order.index(color) + 1
        else:
            priority = len(color_order) + 1

        approach_position = f"{{X {mid_x}, Y {mid_y}, Z {approach_z}, A 0, B 0, C 0}}"
        goal_position = f"{{X {mid_x}, Y {mid_y}, Z {goal_z}, A 0, B 0, C 0}}"
        
        robot.write("APPROACH_POS", approach_position)
        robot.write("GOAL_POS", goal_position)
        robot.write("COLOR", color)
        robot.write("ORIENTATION", orientation)
        robot.write("PRIORITY", str(priority))
        
    except Exception as e:
        print("Error:", e)

######## TKINTER INTERFACE ##########

# Function to handle draw one button click
def draw_one_clicked():
    global selected_draw_type
    selected_draw_type = 1
    try:
        robot.write("DRAW_TYPE", "1")
    except Exception as e:
        print("Error:", e)

# Function to handle draw two button click
def draw_two_clicked():
    global selected_draw_type
    selected_draw_type = 2
    try:
        robot.write("DRAW_TYPE", "2")
    except Exception as e:
        print("Error:", e)

# Function to handle draw three button click
def draw_three_clicked():
    global selected_draw_type
    selected_draw_type = 3
    try:
        robot.write("DRAW_TYPE", "3")
    except Exception as e:
        print("Error:", e)

# Function to create Tkinter interface
def create_interface():
    root = tk.Tk()
    root.title("Robot Controller")
    root.geometry('600x600')

    button1 = tk.Button(root, text="Draw 1", command=draw_one_clicked, width=15, height=2, bd=3, font=("Arial", 12), bg="lightgray", fg="black", padx=10, pady=5)
    button1.pack(padx=20, pady=20)

    button2 = tk.Button(root, text="Draw 2", command=draw_two_clicked, width=15, height=2, bd=3, font=("Arial", 12), bg="lightgray", fg="black", padx=10, pady=5)
    button2.pack(padx=20, pady=20)

    button3 = tk.Button(root, text="Draw 3", command=draw_three_clicked, width=15, height=2, bd=3, font=("Arial", 12), bg="lightgray", fg="black", padx=10, pady=5)
    button3.pack(padx=20, pady=20)

    root.mainloop()
    global video_running
    video_running = False

# Run the video capture in a separate thread
video_thread = threading.Thread(target=video_capture)
video_thread.start()

# Run the Tkinter interface in the main thread
create_interface()
