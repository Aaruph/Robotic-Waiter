from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
import subprocess
import cv2
import numpy as np
import serial
import time
from pyzbar import pyzbar
import threading
import requests
from Speech import announce_order_arrival
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

import pygame

# Initialize pygame mixer
pygame.mixer.init()






FLASK_API_URL = 'http://localhost:5000/api/robot_command'

class SerialThread(threading.Thread):
    def __init__(self):
        super(SerialThread, self).__init__()
        self.lock = threading.Lock()
        self.running = True
        self.command = None
        self.ser  = None
        try:
            self.ser = serial.Serial('COM8', 115200, timeout=.025, write_timeout=.025)  # Adjust to your Arduino's serial port
            time.sleep(2)  # Allow time for serial connection to establish
        except serial.SerialException as e:
            print(f"Error: {e}")
            # exit()
        print("INIT" , self.ser)
    def run(self):
        print("SELF SER ",self.ser)
        while self.running :
            if self.command:
                with self.lock:
                    command_to_send = self.command
                    self.command = None
                try:
                    if self.ser.isOpen():
                        
                        # print(f"Before command: {command_to_send.strip()}")
                        self.ser.write(command_to_send.encode())  # Send the command to the Arduino
                        # print(f"After command: {command_to_send.strip()}")
                        self.ser.flush()  # Ensure all data is sent immediately
                    else:
                        print("Serial port is not open.")
                except serial.SerialException as e:
                    try:
                        self.ser.close()
                        self.ser = serial.Serial('COM8', 115200, timeout=.025, write_timeout=.025)  # Adjust to your Arduino's serial port
                        time.sleep(2)  # Allow time for serial connection to establish
                    except serial.SerialException as e:
                        print(f"Renit Error: {e}")
                    print(f"Serial communication error: {e}")
            time.sleep(0.025)  # Adjust sleep time as needed

    def send_command(self, command):
        with self.lock:
            self.command = command

    def stop(self):
        self.running = False

serial_thread = SerialThread()

# Function to send motor commands
def send_wheel_command(left_speed, right_speed):
    command = f"MOVE {left_speed} {right_speed}\n"
    serial_thread.send_command(command)

def find_path(hsv):
    # Define black color range in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    # Create a binary mask where black colors are white
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to track the widest contour
    max_width = 0
    largest_contour = None
    
    # Iterate through contours to find the one with the maximum width
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > max_width:
            max_width = w
            largest_contour = contour
    
    return largest_contour


def detect_obstacle(hsv):
    # Define color range for obstacle detection (e.g., red obstacles)
    lower_obstacle = np.array([0, 50, 50])
    upper_obstacle = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_obstacle, upper_obstacle)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 200:  # Adjust the area threshold based on obstacle size
            return True
    return False

def play_beep():
    # Load and play the beep sound
    pygame.mixer.music.load('beep2.wav')  # Update with your beep sound file path
    pygame.mixer.music.play()
    print("Beep sound played")
# Function to detect lines and calculate control signals
def detect_lines(image, move):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = find_path(hsv)
    mask = np.zeros_like(hsv[:, :, 0])
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mask = cv2.medianBlur(mask, 5)
    res = cv2.bitwise_and(image, image, mask=mask)
    copy = image.copy()
    line_detected = False
    obstacle_detected = detect_obstacle(hsv)
    
    if obstacle_detected:
        play_beep()  # Play beep sound when obstacle is detected
        print("Obstacle detected")
        cv2.putText(image, 'Obstacle Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        send_wheel_command(0, 0)  # Stop the robot
    elif contour is not None:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.line(image, (cx, 0), (cx, image.shape[0]), (255, 0, 0), 1)
            cv2.line(image, (0, cy), (image.shape[1], cy), (255, 0, 0), 1)
            cv2.drawContours(copy, [contour], -1, (0, 255, 0), 1)
            if not move:
                send_wheel_command(0, 0)
            else:
                if 45 < cx < 80:
                    send_wheel_command(4, 4)  # Move forward
                    print("moving forward")
                # elif cx <= 45:
                #     send_wheel_command(-4, 4)  # Turn left
                #     print("Turning left")
                # elif cx >= 80:
                #     send_wheel_command(4, -4)  # Turn right
                #     print("Turning Right")
            line_detected = True

    return image, copy, hsv, res, line_detected

def announce_order_in_thread():
    threading.Thread(target=announce_order_arrival).start()


def update_order_status_in_thread(table_number):
    try:
        response = requests.post(f"{FLASK_API_URL}/update_order_status", json={"table_number": table_number})
        if response.status_code == 200:
            print(f"Order status for Table no {table_number} updated successfully.")
        else:
            print(f"Failed to update order status for Table no {table_number}. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error updating order status: {e}")

# Function to decode QR codes
def decode_qr_code(frame, table_destination):
    global processing_thread
    qr_codes = pyzbar.decode(frame)
    stop_robot = False
    td = table_destination
    for qr_code in qr_codes:
        (x, y, w, h) = qr_code.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        qr_data = qr_code.data.decode('utf-8')
        qr_type = qr_code.type
        text = f"{qr_data} ({qr_type})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Decoded QR code: {qr_data}")
        if qr_data.startswith("Table no"):
            table_number = qr_data.split(" ")[-1]
            table_destination_number = table_destination.split(" ")[-1]
            print(table_destination_number, table_number)
            if table_number == "0":
                print(f"Detected QR code for Table no 0. Stopping robot completely.")
                processing_thread.move_forward = False
                processing_thread.stop()  # Assuming there's a stop method to halt all movements
                stop_robot = True
                break  # Exit the loop as we need to stop immediately
            elif table_destination_number == table_number:
                print(f"Detected QR code for Table no {table_number}. Stopping robot for 20 seconds.")
                announce_order_in_thread()  # Run announcement in a separate thread
                td = "Table no 0"
                processing_thread.move_forward = False
                stop_robot = True
                update_thread = threading.Thread(target=update_order_status_in_thread, args=(table_number,))
                update_thread.start()
    return frame, stop_robot, td

# Thread class for video capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super(VideoCaptureThread, self).__init__()
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.capture.set(3, 300)
        self.capture.set(4, 300)
        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.stop_time = 0

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.capture.release()

# Thread class for frame processing
class FrameProcessingThread(threading.Thread):
    def __init__(self, video_thread):
        super(FrameProcessingThread, self).__init__()
        self.video_thread = video_thread
        self.running = True
        self.table = "Table no 0"
        self.isatzero = True
        self.move_forward = False
        self.initial_position = None
        self.start_time = None
        self.stop_time = None
        self.travel_time = None
        self.backward_start_time = None
        
    def run(self):
        desired_fps = 10  # Adjust desired frames per second here
        delay = int(1000 / desired_fps)

        while self.running:
            frame = self.video_thread.read()
            if frame is None:
                continue

            # QR code detection
            try:
                frame, stop_robot, self.table = decode_qr_code(frame, self.table)
            except Exception as e:
                print(f"Error in QR code decoding: {e}")
                continue

            # Line following
            cropped = frame[100:250, 90:200]
            original, center, hsv, res, line_detected = detect_lines(cropped, self.move_forward)

            # Display frames
            try:
                cv2.imshow('QR Code Scanner', frame)
                cv2.imshow('hsv', hsv)
                cv2.imshow('res', res)
                if center is not None:
                    cv2.imshow('Detected Lines', center)
                if original is not None:
                    cv2.imshow('Contours', original)
            except Exception as e:
                print(f"Error displaying frames: {e}")

            if stop_robot:
                if self.stop_time is None:
                    self.stop_time = time.time()
                    self.travel_time = self.stop_time - self.start_time
                    print(f"Travel time to destination: {self.travel_time} seconds")

            if self.stop_time is not None:
                elapsed_stop_time = time.time() - self.stop_time

                if elapsed_stop_time < 10:
                    print("Stopping robot...")
                    send_wheel_command(0, 0)
                else:
                    
                    if self.backward_start_time is None:
                        self.backward_start_time = time.time()
                        print("Started backward movement timer.")
                    else:
                        elapsed_backward_time = time.time() - self.backward_start_time
                        if elapsed_backward_time < self.travel_time:
                            print(f"Moving backward... Time elapsed: {elapsed_backward_time} seconds")
                            send_wheel_command(-5, -5)  # Adjust values as needed for backward movement
                        else:
                            print("Stopping at initial position.")
                            send_wheel_command(0, 0)
                            self.stop_time = None  
                            self.backward_start_time = None

            if not line_detected:
                print("No line detected. Sending stop command.")
                send_wheel_command(0, 0)

            if self.start_time is None and self.move_forward:
                self.start_time = time.time()

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                self.running = False
                break

    def stop(self):
        self.running = False

    def set_initial_position(self, position):
        self.initial_position = position




video_thread = VideoCaptureThread()  # Correct initialization
processing_thread = FrameProcessingThread(video_thread)  # Pass video_thread to processing_thread
# serial_thread = SerialThread()
def main():
    print("Robot thread")
    serial_thread.start()  # Start the serial communication thread
    video_thread.start()
    processing_thread.start()

    try:
        while processing_thread.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        video_thread.stop()
        processing_thread.stop()
        serial_thread.stop()

    finally:
        video_thread.stop()
        processing_thread.stop()
        serial_thread.stop()

        video_thread.join()
        processing_thread.join()
        serial_thread.join()
        cv2.destroyAllWindows()
        serial_thread.ser.close()
        print("Serial connection closed.")



app = Flask(__name__)
# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///restaurant.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'supersecretkeythatshouldbechanged'
db = SQLAlchemy(app)



# Define models
class MenuItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    price = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(255), nullable=True)

    
class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('menu_item.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    table_number = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(50), default='Pending')

    # Define the relationship to MenuItem
    item = db.relationship('MenuItem', backref='orders')

# Create the database
with app.app_context():
    db.create_all()

@app.route('/')
def welcome():
    return render_template('customer/welcome.html')

@app.route('/menu', methods=['GET'])
def menu():
    items = MenuItem.query.all()
    return render_template('customer/menu.html', items=items)

@app.route('/order/<int:item_id>', methods=['POST','GET'])
def order(item_id):
    item = MenuItem.query.get_or_404(item_id)
    if item is None:
        return "Item not found", 404

    # Print item details for debugging
    print(f"Item Name: {item.name}")
    print(f"Item Description: {item.description}")
    print(f"Item Price: {item.price}")

    if request.method == 'POST':
        quantity = request.form['quantity']
        table_number = request.form['table_number']
        new_order = Order(item_id=item.id, quantity=quantity, table_number=table_number)
        db.session.add(new_order)
        db.session.commit()
        return redirect(url_for('confirmation'))
    return render_template('customer/order.html',item=item)

@app.route('/add_menu_item', methods=['GET', 'POST'])
def add_menu_item():
    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        price = request.form['price']
        image = request.files['image']

        if image:
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new_item = MenuItem(name=name, description=description, price=price, image_filename=filename)
        else:
            new_item = MenuItem(name=name, description=description, price=price)

        db.session.add(new_item)
        db.session.commit()
        flash('Menu item added successfully!')
        return redirect(url_for('add_menu_item'))
    return render_template('kitchen/add_menu_item.html')

@app.route('/confirmation')
def confirmation():
    return render_template('customer/confirmation.html')
@app.route('/total_orders')
def total_orders():
    orders = Order.query.join(MenuItem).add_columns(MenuItem.name.label('item_name')).all()
    return render_template('kitchen/total_orders.html', orders=orders)

@app.route('/send_command/<int:order_id>', methods=['POST'])
def send_command(order_id):
    order = Order.query.get(order_id)
    print(order.table_number)
    if order:
        # Update order status to 'Processing'
        # order.status = 'Completed'
        # db.session.commit()
        processing_thread.table = "Table no " + str(order.table_number)
        processing_thread.move_forward = True
        video_thread.stop_time = 1
        
        # Send command to the robot with the table number
        # subprocess.Popen(["python", "robot/robot.py", str(order.table_number)])
        
        return redirect(url_for('total_orders'))
    return jsonify({'error': 'Order not found'}), 404

@app.route('/api/robot_command/update_order_status', methods=['POST'])
def update_order_status():
    data = request.get_json()
    table_number = data.get('table_number')
    if table_number:
        order = Order.query.filter_by(table_number=table_number, status='Pending').first()
        if order:
            order.status = 'Completed'
            db.session.commit()
            return jsonify({'message': 'Order status updated successfully'}), 200
        return jsonify({'error': 'Order not found or already completed'}), 404
    return jsonify({'error': 'Invalid table number'}), 400

def run_app():
    app.run(debug=False, threaded=True)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_app)
    flask_thread.start()
    
    serial_thread.start()  # Start the serial communication thread
    video_thread.start()
    processing_thread.start()

    try:
        while processing_thread.is_alive():
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        video_thread.stop()
        processing_thread.stop()
        serial_thread.stop()

    finally:
        video_thread.stop()
        processing_thread.stop()
        serial_thread.stop()

        video_thread.join()
        processing_thread.join()
        serial_thread.join()
        cv2.destroyAllWindows()
        serial_thread.ser.close()
        print("Serial connection closed.")

    # robot_thread = threading.Thread(target=main)
    
    # robot_thread.start()

    # robot_thread.join()
    flask_thread.join()