import tkinter as tk
import cv2
import threading
import numpy as np
from sklearn.cluster import KMeans
import math

# Flag to control the video loop
video_running = True

# Function to find the USB camera index (assuming it is not 0)
def find_usb_camera_index():
    for index in range(1, 10):  # Start from 1 to skip the built-in camera
        vid = cv2.VideoCapture(index)
        if vid.isOpened():
            vid.release()
            return index
    return -1

# Function to draw the bounding box around detected Legos
def draw_bounding_box(frame, circles):
    # Extract the x and y coordinates from the circles
    x_coords = [circle[0] for circle in circles]
    y_coords = [circle[1] for circle in circles]

    # Calculate the top-left and bottom-right coordinates of the bounding box
    top_left = (min(x_coords), min(y_coords))
    bottom_right = (max(x_coords), max(y_coords))

    # Draw the bounding box
    cv2.rectangle(frame, top_left, bottom_right, (128, 128, 128), 2)  # Grey color for the bounding box

    return frame

# Function to determine the orientation of the Lego
def determine_orientation(circles):
    # Calculate the angle between the first two circles and the horizontal axis
    x1, y1 = circles[0][:2]
    x2, y2 = circles[1][:2]

    angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

    # Determine orientation based on the angle with buffer zones
    if -15 <= angle <= 15 or 165 <= angle <= 180 or -180 <= angle <= -165:
        return "Horizontal"
    elif 75 <= angle <= 105 or -105 <= angle <= -75:
        return "Vertical"
    elif 15 < angle < 75:
        return "Diagonal Right"
    elif -75 < angle < -15:
        return "Diagonal Left"
    else:
        return "Unknown"

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

    while video_running:
        ret, frame = vid.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Detect circles in the frame
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=30,
            param1=50, 
            param2=30, 
            minRadius=15, 
            maxRadius=30
        )

        lego_count = 0

        if circles is not None: # If circles are detected
            circles = np.uint16(np.around(circles))

            if len(circles[0]) >= 4:  # Ensure there are enough circles to form Legos
                
                n_clusters = len(circles[0]) // 4
                if n_clusters > 0:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    kmeans.fit(circles[0][:, :2])
                    labels = kmeans.labels_

                    # Draw each group as a separate Lego
                    for i in range(max(labels) + 1):
                        lego_circles = circles[0][labels == i]

                        if len(lego_circles) == 4:
                            lego_count += 1
                            sorted_lego_circles = sorted(lego_circles, key=lambda x: (x[0], x[1])) # Sort by x, then y

                            for (x, y, r) in sorted_lego_circles:
                                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)  # Draw circle
                                cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)  # Draw center

                            # Draw bounding box around the detected Lego
                            frame = draw_bounding_box(frame, sorted_lego_circles)

                            # Calculate the midpoint of the square
                            mid_x = int(sum([coord[0] for coord in sorted_lego_circles]) / 4)
                            mid_y = int(sum([coord[1] for coord in sorted_lego_circles]) / 4)
                            cv2.circle(frame, (mid_x, mid_y), 5, (255, 255, 0), -1)  # Draw midpoint

                            # Determine the orientation
                            orientation = determine_orientation(sorted_lego_circles)
                            print(f"Lego {lego_count} orientation: {orientation}")
                            cv2.putText(frame, orientation, (mid_x, mid_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                            # Extract color information and print the color of the lego
                            lego_color = extract_lego_color(frame, sorted_lego_circles)
                            cv2.putText(frame, lego_color, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
        if lego_count > 0:
            print(f"Total Legos detected: {lego_count}")
            print(f"Lego {lego_count} midpoint coordinates: ({mid_x}, {mid_y})")
            send_midpoint(mid_x, mid_y)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            break

    vid.release()
    cv2.destroyAllWindows()

def send_midpoint(mid_x, mid_y):
    try:
        print(mid_x, mid_y)
    except Exception as e:
        print("Error:", e)

def extract_lego_color(frame, circles):
    for (x, y, r) in circles:
        lego_roi = frame[y-r:y+r, x-r:x+r]
        
    
    # setting values for base colors 
    b = frame[:, :, :1] 
    g = frame[:, :, 1:2] 
    r = frame[:, :, 2:] 
  
    # computing the mean 
    b_mean = np.mean(b) 
    g_mean = np.mean(g) 
    r_mean = np.mean(r) 
  
    # displaying the most prominent color 
    if (b_mean > g_mean and b_mean > r_mean): 
        print("Blue")
        send_color("Blue")
    if (g_mean > r_mean and g_mean > b_mean): 
        print("Green")
        send_color("Green") 
    else: 
        send_color("Red")
        print("Red")
        
# function to send the orientation to the robot
def send_orientation(orientation):
    try:
        if orientation == "vertical":
            print("vertical")
        elif orientation == "horizontal":
            print("horizontal")
        elif orientation == "diagonal right":
            print("diagonal right")
        elif orientation == "diagonal left":
            print("diagonal left")
             
    except Exception as e:
        print("Error:", e)      

# Function to send color to the robot
def send_color(color):
    try:
        if color == "Blue":
            print("Blue")
        elif color == "Green":
            print("Green")
        elif color == "Red":
            print("Red")
    except Exception as e:
        print("Error:", e)

# Function to send flag to the robot
def send_flag(flag):
    try:
        if flag == 1:
            print(1)
        elif flag == 2:
            print(2)
        elif flag == 3:
            print(3)
    except Exception as e:
        print("Error:", e)

# Button click event handlers
def draw_one_clicked():
    print("Button 1")
    send_flag(1)

def draw_two_clicked():
    print("Button 2")
    send_flag(2)

def draw_three_clicked():
    print("Button 3")
    send_flag(3)

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
