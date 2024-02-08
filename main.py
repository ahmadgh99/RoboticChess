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
import arm_module as am
import GUI as ui
import utilites as utils

start_flag = False
ready_flag = False
found_flag = False
pause_flag = False
started_flag = False
end_game_flag = False

Threads = []

def start_btn(UI):
    global start_flag,ready_flag,started_flag
    if not started_flag:
        start_flag = True
        started_flag = True
        UI.add_message("Place the pieces then press ready")
        UI.start_button.text = "Ready"
    else:
        UI.start_button.text = "Running"
        if not ready_flag:
            UI.enable_start_btn()
        ready_flag = True
        
def pause_btn(UI):
    global pause_flag
    if not pause_flag:
        pause_flag = True
        UI.pause_button.text = "Resume"
        UI.add_message("The has been paused")
    else:
        pause_flag = False
        UI.pause_button.text = "Pause"
        UI.add_message("The has been resumed")

def find_board(image_processor,camera,game):
    global found_flag,end_game_flag,pause_flag
    while not found_flag and not end_game_flag:
        if pause_flag:
            continue
        image_processor.img_rgb = camera.capture_frame().copy()
        image_processor.initial_detect()
        wrap_found = image_processor.detect_chessboard()
        game.reinitilize_chessboard()
        found_flag = False if wrap_found is None or len(game.physical_board.squares) == 0 else True
    Threads.remove(threading.current_thread())
    image_processor.save_masks()
    print("Thread 3 Out")

def display_for_debug(image_processor,UI):
    global found_flag,started_flag,game
    frame_to_display = image_processor.warped_frame if found_flag else image_processor.img_rgb
    if  found_flag and not started_flag and UI.start_button.disabled:
        UI.enable_start_btn()
    return frame_to_display


def capture_and_process_thread(image_processor, camera, Game, MD,UI,exec):
    global start_flag,ready_flag,end_game_flag

    while not end_game_flag:
        if pause_flag:
            continue
        frame = camera.capture_frame()  # Capture frame from Realsense
        depth_frame = camera.capture_depth()
        
        image_processor.img_rgb = frame.copy()
        image_processor.detect_chessboard()
        
        if start_flag:
            Game.initialize_empty_board()
            MD.get_background(depth_frame)
            Game.setup_pieces()
            start_flag = False
            Game.change_difficulty(UI.select_difficulty())
        if ready_flag and MD.detect_motion(depth_frame) is False:
            arm_inst = Game.detect_player_move()
            if arm_inst != "":
                exec.execute_actions(arm_inst)
    Threads.remove(threading.current_thread())
    print("Thread 1 Out")
       

def main():
    global found_flag,end_game_flag
    cam = camera.Camera()
    
    arm = am.RoboticArm('192.168.1.162')
    executor = am.ActionExecutor(arm)

    image_processor = ip.ImageProcessor()
    
    MotionDetector = MD.motion_detector()

    cam.config_cam()

    UI = ui.GUI(start_btn,pause_btn)
    
    # Capture and discard frames for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
        _ = cam.capture_frame()

    frame = cam.capture_frame()


    image_processor.img_rgb = frame
    image_processor.initial_detect()
    found = image_processor.detect_chessboard()
    Game = chess.ChessGame(frame,image_processor,UI)
    
    if not found or len(Game.physical_board.squares) == 0:
        Threads.append(threading.Thread(target=find_board,args=(image_processor, cam, Game)))
        Threads[0].start()
    else:
        found_flag = True
        image_processor.save_masks()
        
    proccessor = threading.Thread(target=capture_and_process_thread, args=(image_processor, cam, Game,MotionDetector,UI,executor))
    Threads.append(proccessor)
    proccessor.start()
    
    UI.schedule_update(display_for_debug,image_processor,UI)
    UI.run()
    end_game_flag = True
    for Thread in Threads:
        Thread.join()
    Game.engine_off()
    cam.quit()
    arm.disconnect()
    print("Thread 0 Out")

if __name__ == "__main__":
    main()
