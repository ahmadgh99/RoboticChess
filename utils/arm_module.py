import time
from xarm.wrapper import XArmAPI


class RoboticArm:
    def __init__(self, ip_address, pickup_height=160, movement_speed=50, drop_speed=10):
        self.arm = XArmAPI(ip_address)
        self.arm.connect()
        self.arm.set_mode(0)
        self.arm.set_tcp_maxacc(1000)
        self.pickup_height = pickup_height
        self.movement_speed = movement_speed
        self.drop_speed = drop_speed

    def check_right_pos(self, desired_position):
        while True:
            threshold = 5
            current_position = self.arm.get_position()
            distance_to_target = ((desired_position[0] - current_position[1][0]) ** 2 +
                                  (desired_position[1] - current_position[1][1]) ** 2 +
                                  (desired_position[2] - current_position[1][2]) ** 0.5)
            if distance_to_target < threshold:
                break
            time.sleep(0.1)

    def move_to_height(self, target_height=200):
        current_position = self.arm.get_position()
        self.arm.set_position(current_position[1][0], current_position[1][1], target_height, speed=self.movement_speed)
        self.check_right_pos([current_position[1][0], current_position[1][1], target_height])

    def move_and_pick(self, x, y):
        self.move_to_height()
        self.arm.set_position(x, y, 200, speed=self.movement_speed)
        self.check_right_pos([x, y, 200])
        self.arm.set_position(x, y, self.pickup_height, speed=self.drop_speed)
        self.check_right_pos([x, y, self.pickup_height])
        self.pick()

    def move_and_release(self, x, y):
        self.arm.set_position(x, y, 200, speed=self.movement_speed)
        self.check_right_pos([x, y, 200])
        self.arm.set_position(x, y, self.pickup_height, speed=self.drop_speed)
        self.check_right_pos([x, y, self.pickup_height])
        self.release()

    def pick(self):
        print("Picking up chess piece...")
        self.arm.set_vacuum_gripper(True)

    def release(self):
        print("Releasing the chess piece...")
        self.arm.set_vacuum_gripper(False)

    def disconnect(self):
        self.arm.disconnect()


class Position:
    @staticmethod
    def get_coordinates(position):
        # Assume a mapping for simplicity. In actual, it should read from file or database.
        mapping = {
            'A1': (100, 100),
            'A2': (100, 200),
            # ... add other mappings
        }
        return mapping.get(position, (None, None))


class ActionExecutor:
    def __init__(self, arm):
        self.arm = arm

    def execute_actions(self, actions_file):
        try:
            with open(actions_file, 'r') as file:
                actions = file.readlines()

            for action in actions:
                from_position, to_position = action.strip().split('->')
                from_x, from_y = Position.get_coordinates(from_position.strip())
                to_x, to_y = Position.get_coordinates(to_position.strip())

                if from_x is not None and from_y is not None and to_x is not None and to_y is not None:
                    self.arm.move_and_pick(from_x, from_y)
                    self.arm.move_and_release(to_x, to_y)

        except Exception as e:
            print(f"An error occurred: {e}")


