import sys
sys.path.append('./utils')
sys.path.append('./chess')
import camera
import chess_module as chess
import image_processing as ip
import numpy as np
import threading
import time
import cv2
import motion_detector as MD
from arm_module import RoboticArm as arm

start_flag = False
ready_flag = False

# Usage
#arm = RoboticArm('192.168.1.162')
#executor = ActionExecutor(arm)
#executor.execute_actions('actions.txt')
#arm.disconnect()

def display_for_debug(image_processor):
        while True:
            cv2.imshow("Warped image of the chessboard",image_processor.warped_frame)
            cv2.waitKey(1)

def capture_and_process_thread(image_processor, camera, Game, MD):
    global start_flag,ready_flag

    while True:
        frame = camera.capture_frame()  # Capture frame from Realsense
        depth_frame = camera.capture_depth()
        processed_frame = ip.warp_frame(frame, image_processor.roi_mask, image_processor.cleaned_mask_cc)        # Process the frame
        
        image_processor.img_rgb = frame
        image_processor.warped_frame = processed_frame

        if start_flag:
            Game.initialize_empty_board()
            MD.get_background(depth_frame)
            Game.setup_pieces()
            start_flag = False
        if ready_flag and MD.detect_motion(depth_frame) is False:
            Game.detect_player_move()
            #time.sleep(1)

def user_input_thread(Game):
    global start_flag,ready_flag

    ask_start_game = True

    while True:
        if ask_start_game:
            user_input = input("Do you want to start the game(Y/N)?").strip().lower()
            if user_input == 'y':
                start_flag = True
                ask_start_game = False
            elif user_input == 'n':
                ask_start_game = False
                user_input = input("Type start game whenever you're ready to start").strip().lower()
        elif not start_flag and not ask_start_game:
            
            if user_input == "start game":
                start_flag = True

        elif start_flag and not ready_flag:
            user_input = input("After placing all the pieces type: ready ").strip().lower()
            if user_input == "ready":
                ready_flag = True
        elif user_input == "set depth threshold":
            value = input(f"set a new threshold factor, currently it is {MD.get_depth_factor()}").strip()
            MD.set_threshold(value)

        else:
            time.sleep(1)  # Sleep for a short duration before checking again.

       

def main():

    cam = camera.Camera()
    
    image_processor = ip.ImageProcessor()
    
    MotionDetector = MD.motion_detector()

    cam.config_cam()

    # Capture and discard frames for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
        _ = cam.capture_frame()

    frame = cam.capture_frame()


    image_processor.img_rgb = frame
    
    image_processor.initial_detect()
    image_processor.detect_chessboard()


    Game = chess.ChessGame(frame,image_processor)

    threading.Thread(target=capture_and_process_thread, args=(image_processor, cam, Game,MotionDetector)).start()
    threading.Thread(target=user_input_thread,args=(Game,)).start()
    threading.Thread(target=display_for_debug,args=(image_processor,)).start()
        

if __name__ == "__main__":
    main()
