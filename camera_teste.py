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
# https://pypi.org/project/py-openshowvar/

robot = openshowvar('192.168.1.1', 7000)  # Replace with your robot's IP and port

try:
    robot.can_connect()
    print("Connection to the robot established successfully.")
except Exception as e:
    print(f"Failed to connect to the robot: {e}")

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
    global video_running
    camera_index = find_usb_camera_index()
    if camera_index == -1:
        print("No camera found")
        return

    vid = cv2.VideoCapture(camera_index)
    if not vid.isOpened():
        print("Could not open camera")
        return

    work_area_defined = False
    top_left_square = None
    pixel_to_cm_ratio = None

    while video_running:
        ret, frame = vid.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Detect circles in the frame
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

            if len(circles[0]) == 4:  # Ensure there are enough circles to form Legos
                n_clusters = len(circles[0]) // 4
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    kmeans.fit(circles[0][:, :2])
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

                            # Determine the orientation
                            orientation = determine_orientation(sorted_lego_circles)
                            print(f"Lego {lego_count} orientation: {orientation}")
                            cv2.putText(frame, orientation, (mid_x, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Write the orientation on the frame

                            # Extract color information and print the color of the lego
                            lego_color = extract_lego_color(frame)
                            cv2.putText(frame, lego_color, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # Write the color on the frame

                            if work_area_defined:
                                # Transform coordinates from pixels to cm
                                mid_x_cm, mid_y_cm = pixel_to_mm(mid_x, mid_y, top_left_square, pixel_to_mm_ratio)
                                send_robot_approx(mid_x_cm, mid_y_cm, lego_color, orientation)

        # black squares : work area
        if not work_area_defined:
            # Detect black squares to define the work area
            black_squares = detect_black_squares(blurred)

            if len(black_squares) == 3:
                # Define the work area
                top_left_square = min(black_squares, key=lambda x: (x[0], x[1]))
                top_right_square = min(black_squares, key=lambda x: (x[0], -x[1]))
                bottom_left_square = min(black_squares, key=lambda x: (-x[0], x[1]))

                # Calculate the pixel-to-cm ratio
                distance_px = np.linalg.norm(np.array(top_left_square[:2]) - np.array(top_right_square[:2]))
                distance_cm = 10  # Assuming the distance between squares is 10 cm
                pixel_to_mm_ratio = distance_cm / distance_px

                work_area_defined = True
                print("Work area defined.")

        if lego_count > 0:
            print(f"Total Legos detected: {lego_count}")

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            break

    vid.release()
    cv2.destroyAllWindows()


# Function to detect black squares
def detect_black_squares(blurred):
    black_squares = []
    # Use thresholding to detect black squares
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > 100:
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                black_squares.append((cX, cY))

    return black_squares

# Function to convert pixels to cm
def pixel_to_mm(x, y, origin, ratio):
    x_mm = (x - origin[0]) * ratio * 10  # Convert cm to mm
    y_mm = (y - origin[1]) * ratio * 10  # Convert cm to mm
    return x_mm, y_mm


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
        return "B" #blue
    elif (g_mean > r_mean and g_mean > b_mean):
        return "G" #green
    else:
        return "R" #red

######## Robot Communication ##########

# Function to send a flag to the robot
def send_flag(flag):
    try:
       
        robot.write("F", str(flag)) # flag
    except Exception as e:
        print("Error:", e)

# Function to send approximation coordinates to the robot
# Function to send approximation coordinates to the robot
def send_robot_approx(mid_x, mid_y, color, orientation):
    try:
        # Lego height
        lego_height = 4  # lego height in cm
        lego_height_mm = lego_height * 10  # Convert cm to mm

        # Define approach height above the Lego's top surface
        approach_height_above_lego = 1  # Adjust this value as needed
        approach_height_above_lego_mm = approach_height_above_lego * 10  # Convert cm to mm

        # Calculate approach height in mm
        approach_z = -(lego_height_mm + approach_height_above_lego_mm)

        goal_z = 0

        # Send the coordinates to the robot based on the color
        if color == "R":
            # Forming approach and goal positions
            approach_position_Red = f"{{X {mid_x}, Y {mid_y}, Z {approach_z}, A 0, B 0, C 0}}"  # valores do ABC nao sei
            goal_position_Red = f"{{X {mid_x}, Y {mid_y}, Z {goal_z}, A 0, B 0, C 0}}"

            # Sending coordinates, orientation, and priority to the robot
            robot.write("APPROACH_POS_RED_LEGO", approach_position_Red)
            robot.write("GOAL_POS_RED_LEGO", goal_position_Red)
            robot.write("C", color)
            robot.write("O", orientation)

        elif color == "G":
            approach_position_Green = f"{{X {mid_x}, Y {mid_y}, Z {approach_z}, A 0, B 0, C 0}}"  # valores do ABC nao sei
            goal_position_Green = f"{{X {mid_x}, Y {mid_y}, Z {goal_z}, A 0, B 0, C 0}}"

            # Sending coordinates, orientation, and priority to the robot
            robot.write("APPROACH_POS_GREEN_LEGO", approach_position_Green)
            robot.write("GOAL_POS_GREEN_LEGO", goal_position_Green)
            robot.write("C", color)
            robot.write("O", orientation)

        elif color == "B":
            approach_position_Blue = f"{{X {mid_x}, Y {mid_y}, Z {approach_z}, A 0, B 0, C 0}}"  # valores do ABC nao sei
            goal_position_Blue = f"{{X {mid_x}, Y {mid_y}, Z {goal_z}, A 0, B 0, C 0}}"

            # Sending coordinates, orientation, and priority to the robot
            robot.write("APPROACH_POS_BLUE_LEGO", approach_position_Blue)
            robot.write("GOAL_POS_BLUE_LEGO", goal_position_Blue)
            robot.write("C", color)
            robot.write("O", orientation)

    except Exception as e:
        print("Error:", e)


### INTERFACE ###

# Button click event handlers
def draw_one_clicked():
    print("Button 1")
    send_flag(1)  # Send flag to the robot

def draw_two_clicked():
    print("Button 2")
    send_flag(2)  # Send flag to the robot

def draw_three_clicked():
    print("Button 3")
    send_flag(3)  # Send flag to the robot

# Function to create and display the Tkinter interface
def create_interface():
    root = tk.Tk()
    root.title("Robot Controller")
    root.geometry('600x600')

    button1 = tk.Button(root,
                        text="Draw 1",
                        command=draw_one_clicked,
                        width=15,
                        height=2,
                        bd=3,
                        font=("Arial", 12),
                        bg="lightgray",
                        fg="black",
                        padx=10,
                        pady=5)
    button1.pack(padx=20, pady=20)

    button2 = tk.Button(root,
                        text="Draw 2",
                        command=draw_two_clicked,
                        width=15,
                        height=2,
                        bd=3,
                        font=("Arial", 12),
                        bg="lightgray",
                        fg="black",
                        padx=10,
                        pady=5)
    button2.pack(padx=20, pady=20)

    button3 = tk.Button(root,
                        text="Draw 3",
                        command=draw_three_clicked,
                        width=15,
                        height=2,
                        bd=3,
                        font=("Arial", 12),
                        bg="lightgray",
                        fg="black",
                        padx=10,
                        pady=5)
    button3.pack(padx=20, pady=20)
    
    
    
    
    root.mainloop()
    global video_running
    video_running = False  # Stop the video capture loop when Tkinter window closes

# Create threads for video capture and Tkinter interface
video_thread = threading.Thread(target=video_capture)
interface_thread = threading.Thread(target=create_interface)

# Start the threads
video_thread.start()
interface_thread.start()

# Wait for the threads to finish
video_thread.join()
interface_thread.join()
