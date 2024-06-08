import os
import tkinter as tk
import cv2
import threading
import numpy as np
from sklearn.cluster import KMeans
import math

# Suppress joblib warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def determine_orientation(circles):
    x1, y1 = circles[0][:2]
    x2, y2 = circles[1][:2]

    try:
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi
    except ZeroDivisionError:
        return "Undefined"

    angle = (angle + 180) % 360 - 180

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

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not open or find the image")
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Add visual feedback for debugging
    cv2.imshow("Gray Image", gray)
    cv2.waitKey(0)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Add visual feedback for debugging
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)
    
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Add visual feedback for debugging
    cv2.imshow("Adaptive Threshold Image", adaptive_thresh)
    cv2.waitKey(0)
    
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

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow("Detected Circles", frame)
        cv2.waitKey(0)
    else:
        print("No circles were found")
    
    cv2.destroyAllWindows()

def send_midpoint(mid_x, mid_y):
    try:
        print(mid_x, mid_y)
    except Exception as e:
        print("Error:", e)

def extract_lego_color(frame):
    b = frame[:, :, :1]
    g = frame[:, :, 1:2]
    r = frame[:, :, 2:]

    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    if b_mean > g_mean and b_mean > r_mean:
        print("Blue")
        send_color("Blue")
        return "Blue"
    elif g_mean > r_mean and g_mean > b_mean:
        print("Green")
        send_color("Green")
        return "Green"
    else:
        send_color("Red")
        print("Red")
        return "Red"

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

def draw_one_clicked():
    print("Button 1")
    send_flag(1)

def draw_two_clicked():
    print("Button 2")
    send_flag(2)

def draw_three_clicked():
    print("Button 3")
    send_flag(3)

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

image_path = "WIN_20240608_09_40_02_Pro.jpg"

image_thread = threading.Thread(target=process_image, args=(image_path,))
interface_thread = threading.Thread(target=create_interface)

try:
    image_thread.start()
    interface_thread.start()

    image_thread.join()
    interface_thread.join()
except KeyboardInterrupt:
    print("Process interrupted. Exiting...")
finally:
    image_thread.join(timeout=1)
    interface_thread.join(timeout=1)
