import time
import math
from xarm.wrapper import XArmAPI


class RoboticArm:
    def __init__(self, ip_address, pickup_height=78, movement_speed=90,
                drop_speed=50,speed_adjust_distance=20,height=200,board_height=3.5):
        self.arm = XArmAPI(ip_address)
        self.arm.connect()
        self.arm.set_mode(0)
        self.arm.set_tcp_maxacc(1000)
        self.pickup_height = pickup_height
        self.movement_speed = movement_speed
        self.drop_speed = drop_speed
        self.height = height
        self.speed_adjust_distance = speed_adjust_distance
        self.board_height = board_height

    def move_to_height(self):
        current_position = self.arm.get_position()
        self.arm.set_position(current_position[1][0], current_position[1][1], self.height, speed=self.movement_speed,mvacc=200,wait=True,timeout=30)
        self.handle_error_code_24(current_position[1][0], current_position[1][1], self.height)
        
    def check_pickup_success(self):
        # Get the state of the vacuum gripper
        code, state = self.arm.get_vacuum_gripper()

        # Check if the code is 0 (indicating success) and state is 1 (suction cup is on)
        if code == 0 and state == 1:
            return True  # Pickup was successful
        else:
            return False  # Pickup failed

    def move_and_pick(self, x, y):
        self.move_to_height()
        
        self.arm.set_position(x, y, self.pickup_height+self.speed_adjust_distance, speed=self.movement_speed,mvacc=400,wait=True,timeout=30)
        self.handle_error_code_24(x, y,self.pickup_height+self.speed_adjust_distance)
        
        self.arm.set_position(x, y, self.pickup_height, speed=self.drop_speed,mvacc=400,wait=True,timeout=30)
        self.handle_error_code_24(x, y,self.pickup_height)
        
        self.pick()
        self.move_to_height()
        
        pickup_success = self.check_pickup_success()
        
        if not pickup_success:
            self.retry_pickup(x, y)

    def move_and_release(self, x, y,capture):
    
        self.move_to_height()
        
        self.arm.set_position(x, y, self.height, speed=self.movement_speed,mvacc=400,wait=True,timeout=30)
        self.handle_error_code_24(x, y,self.pickup_height)
        
        self.arm.set_position(x, y, self.pickup_height+self.speed_adjust_distance, speed=self.movement_speed,mvacc=400,wait=True,timeout=30)
        self.handle_error_code_24(x, y,self.pickup_height+self.speed_adjust_distance)
        
        if capture is True:
            self.arm.set_position(x, y, self.pickup_height-self.board_height,speed=self.drop_speed,mvacc=400,wait=True,timeout=30)
            self.handle_error_code_24(x, y,self.pickup_height-self.board_height)
        else:
            self.arm.set_position(x, y, self.pickup_height,speed=self.drop_speed,mvacc=400,wait=True,timeout=30)
            self.handle_error_code_24(x, y,self.pickup_height)
            
        self.release()
        self.move_to_height()
        
    def retry_pickup(self, x, y):
        initial_radius = 5
        max_radius = 18
        min_radius = 2
        radius_change_step = 2
        current_radius = initial_radius

        while current_radius <= max_radius:
            retry_offsets = self.calculate_retry_offsets(current_radius)

            for dx, dy in retry_offsets:
                self.arm.set_position(x + dx, y + dy, self.pickup_height + self.speed_adjust_distance, speed=self.movement_speed, mvacc=400, wait=True, timeout=30)
                self.arm.set_position(x + dx, y + dy, self.pickup_height, speed=self.drop_speed, mvacc=400, wait=True, timeout=30)
                
                self.pick()
                
                self.arm.set_position(x + dx, y + dy, self.pickup_height + self.speed_adjust_distance, speed=self.movement_speed, mvacc=400, wait=True, timeout=30)
                
                if self.check_pickup_success():
                    return  # Successful pickup

            # Increase radius for next attempt or decrease if max reached
            if current_radius == max_radius:
                radius_change_step = -1  # Start decreasing the radius
            current_radius += radius_change_step

            # Break if minimum radius is reached and unsuccessful
            if current_radius < min_radius:
                print("Failed to pick up the piece after exhaustive retries.")
                break

    def calculate_retry_offsets(self, radius):
        offset_factor = radius * math.sqrt(2) / 2
        return [
            (radius, 0), (offset_factor, offset_factor), (0, radius), (-offset_factor, offset_factor),
            (-radius, 0), (-offset_factor, -offset_factor), (0, -radius), (offset_factor, -offset_factor)
        ]
        
    def handle_error_code_24(self,x,y,z):
        # Check if the current error is Error Code 24
        if self.arm.error_code == 24:
            print("Error Code 24 detected. Attempting to resolve...")

            # Clear the error
            self.arm.clean_error()
            self.arm.motion_enable(True)
            self.arm.set_state(state=0)
            time.sleep(0.5)

            # Adjust the arm's position
            success = self.adjust_position_after_error()

            if success:
                print("Error resolved. Continuing operation.")
                self.arm.set_position(x, y, z, wait=True, speed = self.drop_speed,mvacc=400,timeout=30)
            else:
                print("Unable to resolve the error. Manual intervention may be required.")

    def adjust_position_after_error(self):
        # Define retry offsets (5mm radius, moving anti-clockwise)
        retry_offsets = [(5, 0), (0, 5), (-5, 0), (0, -5)]
        for dx, dy in retry_offsets:
            # Get the current position
            _, position_list = self.arm.get_position()  # Assuming the second element is the position list [x, y, z, ...]

            # Create a new position by adding offsets to x and y
            new_x = position_list[0] + dx
            new_y = position_list[1] + dy
            new_position = [new_x, new_y, position_list[2]]  # Assuming z is at index 2
            self.arm.set_position(*new_position, wait=True, speed = self.drop_speed,mvacc=400,timeout=30)

            # Check if the error occurs again
            if self.arm.error_code != 24:
                return True  # Error resolved
        
    def move_to_initial_position(self):
        self.arm.set_position(90, 0, 154.2, speed = 70,mvacc=400,wait=True,timeout=30)

    def pick(self):
        print("Picking up chess piece...")
        self.arm.set_vacuum_gripper(True)
        time.sleep(1)

    def release(self):
        print("Releasing the chess piece...")
        self.arm.set_vacuum_gripper(False)
        time.sleep(1)

    def disconnect(self):
        self.arm.disconnect()
    

class Position:
    # Class attributes
    capture_position = [None, None]  # Initialize as None, will set in class method
    capture_offset = 35  # 3 cm in mm
    x_count = 0  # Counter for how many times "X" is received
    
    @classmethod
    def initialize_capture_position(cls):
        cls.h1_x, cls.h1_y = cls.get_coordinates('H1')
        cls.capture_position = [cls.h1_x, cls.h1_y - 50]
        cls.x_count = 0  # Reset the count whenever initializing

    @classmethod
    def update_capture_position(cls):
        if cls.x_count % 2 == 0:
            cls.capture_position[1] -= cls.capture_offset  # Move in negative Y direction
        else:
            cls.capture_position[0] += cls.capture_offset  # Move in positive X direction
            cls.capture_position[1] += cls.capture_offset  # Move in negative Y direction
        cls.x_count += 1
        
    @staticmethod
    def get_coordinates(position):
        # Assume a mapping for simplicity. In actual, it should read from file or database.
        if position != "X":
            mapping = {
                'H1': (125.7, -131.4),
                'H2': (163.7, -131.4),
                'H3': (201.7, -131.4),
                'H4': (239.7, -131.4),
                'H5': (277.7, -131.4),
                'H6': (315.7, -131.4),
                'H7': (353.7, -131.4),
                'H8': (391.7, -131.4),
                'G1': (125.7, -93.4),
                'G2': (163.7, -93.4),
                'G3': (201.7, -93.4),
                'G4': (239.7, -93.4),
                'G5': (277.7, -93.4),
                'G6': (315.7, -93.4),
                'G7': (353.7, -93.4),
                'G8': (391.7, -93.4),
                'F1': (125.7, -55.4),
                'F2': (163.7, -55.4),
                'F3': (201.7, -55.4),
                'F4': (239.7, -55.4),
                'F5': (277.7, -55.4),
                'F6': (315.7, -55.4),
                'F7': (353.7, -55.4),
                'F8': (391.7, -55.4),
                'E1': (125.7, -17.4),
                'E2': (163.7, -17.4),
                'E3': (201.7, -17.4),
                'E4': (239.7, -17.4),
                'E5': (277.7, -17.4),
                'E6': (315.7, -17.4),
                'E7': (353.7, -17.4),
                'E8': (391.7, -17.4),
                'D1': (125.7, 20.6),
                'D2': (163.7, 20.6),
                'D3': (201.7, 20.6),
                'D4': (239.7, 20.6),
                'D5': (277.7, 20.6),
                'D6': (315.7, 20.6),
                'D7': (353.7, 20.6),
                'D8': (391.7, 20.6),
                'C1': (125.7, 58.6),
                'C2': (163.7, 58.6),
                'C3': (201.7, 58.6),
                'C4': (239.7, 58.6),
                'C5': (277.7, 58.6),
                'C6': (315.7, 58.6),
                'C7': (353.7, 58.6),
                'C8': (391.7, 58.6),
                'B1': (125.7, 96.6),
                'B2': (163.7, 96.6),
                'B3': (201.7, 96.6),
                'B4': (239.7, 96.6),
                'B5': (277.7, 96.6),
                'B6': (315.7, 96.6),
                'B7': (353.7, 96.6),
                'B8': (391.7, 96.6),
                'A1': (125.7, 134.6),
                'A2': (163.7, 134.6),
                'A3': (201.7, 134.6),
                'A4': (239.7, 134.6),
                'A5': (277.7, 134.6),
                'A6': (315.7, 134.6),
                'A7': (353.7, 134.6),
                'A8': (391.7, 134.6),
            }
            return mapping.get(position, (None, None))
        else:
             # Handling capture move
            current_capture_pos = Position.capture_position.copy()
            Position.update_capture_position()
            return current_capture_pos


class ActionExecutor:
    def __init__(self, arm):
    
        self.arm = arm
        Position.initialize_capture_position()
        
    def execute_actions(self, actions_string):
            try:
                actions = actions_string.strip().split('\n')
                for action in actions:

                    from_position, to_position = action.strip().split('->')
                    from_x, from_y = Position.get_coordinates(from_position.strip())
                    to_x, to_y = Position.get_coordinates(to_position.strip())

                    if from_x is not None and from_y is not None and to_x is not None and to_y is not None:
                        self.arm.move_and_pick(from_x, from_y)
                        self.arm.move_and_release(to_x, to_y,True if to_position.strip() == "X" else False)
                self.arm.move_to_initial_position()
            except Exception as e:
                print(f"An error occurred: {e}")


