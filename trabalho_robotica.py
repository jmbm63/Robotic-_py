import tkinter as tk
import cv2
import threading
import numpy as np
from sklearn.cluster import KMeans
import math
from py_openshowvar import openshowvar

# Global variables for video loop control
video_running = True
process_frame = False  # Flag to indicate when to process a frame

# Global variables for Lego processing
top_left_square = None
pixel_to_mm_ratio_value = 0  # Renamed variable = 0
lego_count = 0
lego_color = "Unknown"
orientation = "Unknown"
mid_x = 0
mid_y = 0
lego_data = []  # Global variable for Lego data storage


# Robot communication setup
robot = openshowvar('192.168.1.1', 7000)  # Replace with your robot's IP and port

try:
    robot.can_connect()
    print("Connection to the robot established successfully.")
except Exception as e:
    print(f"Failed to connect to the robot: {e}")



# Function to print Lego information
def call_Print(lego_count, lego_color, orientation, mid_x, mid_y):
    print("Lego Count:", lego_count)
    print("Lego Color:", lego_color)
    print("Lego Orientation:", orientation)
    print("Lego Coordinates:", mid_x, mid_y)

# Function to determine the orientation of Lego blocks
def determine_orientation(circles):
    if len(circles) < 2:
        return "Undefined"

    x1, y1 = circles[0][:2]
    x2, y2 = circles[1][:2]

    try:
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    except ZeroDivisionError:
        return "Undefined"
    except OverflowError:
        return "Overflow"

    angle = (angle + 180) % 360 - 180

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

# Function to find USB camera index
def find_usb_camera_index():
    for index in range(1, 10):  # Start from 1 to skip the built-in camera
        vid = cv2.VideoCapture(index)
        if vid.isOpened():
            vid.release()
            return index
    return -1

# Function to capture video
def video_capture():
    global video_running, process_frame
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

        if process_frame:
            process_frame_data(frame)
            process_frame = False

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_running = False
            break

    vid.release()
    cv2.destroyAllWindows()

# Function to process each frame
def process_frame_data(frame):
    global top_left_square, pixel_to_mm_ratio_value, lego_count, lego_color, orientation, mid_x, mid_y, lego_data

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    gray = cv2.medianBlur(gray, 5)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    black_squares = detect_black_squares(blurred)
    if black_squares:
        top_left_square = black_squares[0]
        pixel_to_mm_ratio_value = calculate_pixel_to_mm_ratio()  # Adjust this based on your calibration

        for square in black_squares:
            cv2.rectangle(frame, (square[0] - 5, square[1] - 5), (square[0] + 5, square[1] + 5), (0, 0, 255), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,
            param1=30,
            param2=20,
            minRadius=10,
            maxRadius=35
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))

            # Filter circles based on size
            filtered_circles = []
            for circle in circles[0, :]:
                x, y, r = circle
                if 10 <= r <= 35:  # Radius range for Lego blocks
                    filtered_circles.append(circle)
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

            if len(filtered_circles) >= 4:
                
                kmeans_data = np.array([(circle[0], circle[1]) for circle in filtered_circles])
                kmeans = KMeans(n_clusters=len(filtered_circles) // 4)
                kmeans.fit(kmeans_data)
                centers = kmeans.cluster_centers_

                lego_data = []  # Reset lego_data list

                for center in centers:
                    mid_x, mid_y = center

                    if top_left_square is not None and pixel_to_mm_ratio_value is not None:
                        mid_x_mm, mid_y_mm = pixel_to_mm(mid_x, mid_y, top_left_square, pixel_to_mm_ratio_value)
                        lego_color = extract_lego_color(frame, int(mid_x), int(mid_y))
                        orientation = determine_orientation(filtered_circles)

                        lego_data.append({
                            "mid_x": mid_x_mm,
                            "mid_y": mid_y_mm,
                            "color": lego_color,
                            "orientation": orientation
                        })

                        #print(f"Lego Center Coordinates (mm): {mid_x_mm:.2f}, {mid_y_mm:.2f}")
                        

                        cv2.circle(frame, (int(mid_x), int(mid_y)), 10, (255, 0, 0), -1)

                lego_count = len(lego_data)
                
                
                # Print all Lego data
                for lego in lego_data: #Iterate through all detected Lego blocks and send to the robot
                    
                    call_Print(lego_count, lego["color"], lego["orientation"], lego["mid_x"], lego["mid_y"]) #print in console
                    
                    send_robot_approx(lego["color"], lego["orientation"], lego["mid_x"], lego["mid_y"]) #send to robot
    
    cv2.imshow('processed_frame', frame)
    cv2.waitKey(1)  # Add a small delay to update the GUI


# Function to detect black squares
def detect_black_squares(blurred):
    black_squares = []
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV) 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        
        if len(approx) == 4 and cv2.contourArea(approx) > 100: # 4 cantos e area maior que 100 faz calculo do centro
            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                black_squares.append((cX, cY))

    return black_squares

# Function to extract the color of the Lego block
def extract_lego_color(frame, x, y, radius=30):
    x_start = max(0, x - radius)
    y_start = max(0, y - radius)
    x_end = min(frame.shape[1], x + radius)
    y_end = min(frame.shape[0], y + radius)

    roi = frame[y_start:y_end, x_start:x_end]

    b_mean = np.mean(roi[:, :, 0])
    g_mean = np.mean(roi[:, :, 1])
    r_mean = np.mean(roi[:, :, 2])

    if b_mean > g_mean and b_mean > r_mean:
        return "B"  # Blue
    elif g_mean > r_mean and g_mean > b_mean:
        return "G"  # Green
    else:
        return "R"  # Red


# Function to convert pixel coordinates to mm
def pixel_to_mm(x, y, origin, ratio):
    x_mm = (x - origin[0]) * ratio   
    y_mm = (y - origin[1]) * ratio   
    return x_mm, y_mm

#   calculate the pixel to mm ratio
def calculate_pixel_to_mm_ratio():
    
    lego_length_pixels = 200  # Adjust this based on your calibration
    lego_length_mm = 130  # LEGO piece length in mm (13 cm)
    pixel_to_mm_ratio = lego_length_mm / lego_length_pixels
    return pixel_to_mm_ratio

# Function to send a flag to the robot
def send_flag(flag):
    try:
        robot.write("F", str(flag))
    except Exception as e:
        print("Error:", e)


# Function to send approximation coordinates to the robot
def send_robot_approx( lego_color, orientation_processed, mid_x, mid_y):
    try:
            mid_x_mm = mid_x
            mid_y_mm = mid_y
            color = lego_color
            orientation = orientation_processed

            
            print(" \n \nAqui Lego Color:", lego_color)
            print("Aqui Lego Orientation:", orientation)
            print("Aqui Lego Coordinates:", mid_x, mid_y)
                   
                   
            lego_height = 4
            lego_height_mm = lego_height * 10

            approach_height_above_lego = 1
            approach_height_above_lego_mm = approach_height_above_lego * 10

            approach_z = -(lego_height_mm + approach_height_above_lego_mm)
            goal_z = 0

            if color == "R":
                
                #print("ENTREI AQUI NO VERMELHO")
                approach_position_Red = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {approach_z}, A 0, B 0, C 0}}"
                goal_position_Red = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {goal_z}, A 0, B 0, C 0}}"
                
                print("################## RED ROBOT SEND #########################")
                print("APPROACH_POS_RED_LEGO",approach_position_Red)
                print("GOAL_POS_RED_LEGO",goal_position_Red)
                print("C", color)
                print("O", orientation)
                print("\n\n")
                
                robot.write("APPROACH_POS_RED_LEGO", approach_position_Red)
                robot.write("GOAL_POS_RED_LEGO", goal_position_Red)
                robot.write("C", color)
                robot.write("O", orientation)
                
                
                

            elif color == "G":
                
                #print("ENTREI AQUI NO VERDE")
                approach_position_Green = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {approach_z}, A 0, B 0, C 0}}"
                goal_position_Green = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {goal_z}, A 0, B 0, C 0}}"
                
                
                print("################## Green ROBOT SEND #########################")
                print("APPROACH_POS_GREEN_LEGO",approach_position_Green)
                print("GOAL_POS_GREEN_LEGO",goal_position_Green)
                print("C", color)
                print("O", orientation)
                print("\n\n")
                robot.write("APPROACH_POS_GREEN_LEGO", approach_position_Green)
                robot.write("GOAL_POS_GREEN_LEGO", goal_position_Green)
                robot.write("C", color)
                robot.write("O", orientation)
                
                
                

            elif color == "B":
                
                #print("ENTREI AQUI NO AZUL")
                approach_position_Blue = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {approach_z}, A 0, B 0, C 0}}"
                goal_position_Blue = f"{{X {mid_x_mm}, Y {mid_y_mm}, Z {goal_z}, A 0, B 0, C 0}}"
                
                print("################## Blue ROBOT SEND #########################")
                print("APPROACH_POS_BLUE_LEGO",approach_position_Blue)
                print("GOAL_POS_BLUE_LEGO",goal_position_Blue)
                print("C", color)
                print("O", orientation)
                print("\n\n")
                
                robot.write("APPROACH_POS_BLUE_LEGO", approach_position_Blue)
                robot.write("GOAL_POS_BLUE_LEGO", goal_position_Blue)
                robot.write("C", color)
                robot.write("O", orientation)

                


    except Exception as e:
        print(f"Error: {e}")
        
# Tkinter interface functions
def draw_one_clicked():
    print("Button 1")
    send_flag(1)

def draw_two_clicked():
    print("Button 2")
    send_flag(2)

def draw_three_clicked():
    print("Button 3")
    send_flag(3)

def capture_frame():
    global process_frame
    process_frame = True
    

def create_interface():
    root = tk.Tk()
    root.title("Robot Controller")
    root.geometry('600x600')

    button1 = tk.Button(root, text="Draw 1", command=draw_one_clicked, width=15, height=2, bd=3, font=("Arial", 12),
                        bg="lightgray", fg="black", padx=10, pady=5)
    button1.pack(padx=20, pady=20)

    button2 = tk.Button(root, text="Draw 2", command=draw_two_clicked, width=15, height=2, bd=3, font=("Arial", 12),
                        bg="lightgray", fg="black", padx=10, pady=5)
    button2.pack(padx=20, pady=20)

    button3 = tk.Button(root, text="Draw 3", command=draw_three_clicked, width=15, height=2, bd=3, font=("Arial", 12),
                        bg="lightgray", fg="black", padx=10, pady=5)
    button3.pack(padx=20, pady=20)

    button_frame = tk.Button(root, text="Capture Frame", command=capture_frame, width=15, height=2, bd=3,
                             font=("Arial", 12), bg="lightgray", fg="black", padx=10, pady=5)
    button_frame.pack(padx=20, pady=20)

    root.mainloop()
    global video_running
    video_running = False

# Threads for video capture and interface
video_thread = threading.Thread(target=video_capture)
interface_thread = threading.Thread(target=create_interface)

# Start threads
video_thread.start()
interface_thread.start()

# Wait for threads to finish
video_thread.join()
interface_thread.join()
