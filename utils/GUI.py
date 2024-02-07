import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.core.image import Image as CoreImage
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
import cv2
from kivy.config import Config
from threading import Event
from kivy.graphics import Color, Rectangle
from kivy.uix.modalview import ModalView
import chess_viz as cviz
from PIL import Image as PilImage

symbols_folder = "/Users/ahmadghanayem/Desktop/Technion/Project 2/chess/PIECES"

class GUI():

    class KivyApp(App):
        def __init__(self, gui):
            self.gui = gui
            super().__init__()

        def build(self):
            # Return the root widget of the KivyApp, which is the main layout of the GUI
            return self.gui.main_layout
            
    def __init__(self,start_btn_action,pause_btn_action):
        # Initialize the main layout as a horizontal BoxLayout
        self.main_layout = BoxLayout(orientation='horizontal')
    
        # Left horizontal layout for video and buttons
        left_layout = BoxLayout(orientation='vertical', size_hint_x=None, height=570,width = 500)
        # Right horizontal layout for video and buttons
        right_layout = BoxLayout(orientation='vertical', size_hint_x=None, width=300)
        
        with right_layout.canvas.before:
            Color(1, 0, 0, 1)  # Light Grey color
        
        # Video feed section
        self.image = Image(size_hint=(None, None), size=(500, 500))
        left_layout.add_widget(self.image)
        
        # Create an Image widget to display the chessboard
        self.board_viz = Image()
        right_layout.add_widget(self.board_viz)
        
        # Load and display the chessboard image
        self.display_chessboard_image("")

        # Button section
        button_layout = BoxLayout(orientation='vertical', size_hint_y=1, width=300, spacing=5)
        self.start_button = Button(on_press=lambda instance: start_btn_action(self),text="Start", size_hint_y=None, height=70)
        self.stop_button = Button(on_press=lambda instance: self.quit(),text="Quit", size_hint_y=None, height=70)
        self.pause_button = Button(on_press=lambda instance: pause_btn_action(self),text="Pause", size_hint_y=None, height=70)
        self.give_up_button = Button(text="Give up", size_hint_y=None, height=70)
        button_layout.add_widget(self.start_button)
        button_layout.add_widget(self.pause_button)
        button_layout.add_widget(self.give_up_button)
        button_layout.add_widget(self.stop_button)
        
        # Disable the button and change its background color
        self.start_button.disabled = True
        self.start_button.background_color = [0.5, 0.5, 0.5, 1]  # Grey color
        self.give_up_button.disabled = True
        self.give_up_button.background_color = [0.5, 0.5, 0.5, 1]  # Grey color

        right_layout.add_widget(button_layout)

        # Message system section
        self.messages = BoxLayout(orientation='vertical', spacing=5, size_hint=(None, None), width=500)
        with self.messages.canvas.before:
            Color(0.9, 0.9, 0.9, 1)
            self.msg_rect = Rectangle(size=self.messages.size, pos=self.messages.pos)

        self.message_scroll = ScrollView(size_hint=(1, None), size=(self.main_layout.width, 100))
        self.message_scroll.add_widget(self.messages)
        left_layout.add_widget(self.message_scroll)
        
        # Add the top layout to the main layout
        self.main_layout.add_widget(left_layout)
        self.main_layout.add_widget(right_layout)

        # Create an instance of the internal KivyApp
        self.kivy_app = self.KivyApp(self)

    def quit(self):
        App.get_running_app().stop()
        
    def enable_start_btn(self):
        if self.start_button.disabled:
            self.start_button.disabled = False
            self.start_button.background_color = [1, 1, 1, 1]  # Black color
        else:
            self.start_button.disabled = True
            self.start_button.background_color = [0.5, 0.5, 0.5, 1]  # Grey color
    
    def run(self):
    # Start the internal KivyApp
        self.kivy_app.run()
        
    def update_video_size(self, new_size):
        self.image.size = new_size

    def add_message(self, text):
        # Schedule the _update_gui method to be called on the main thread
        Clock.schedule_once(lambda dt: self._update_gui(text), 0)

    def _update_gui(self, text):
        message = Label(text=text, color=(0, 0, 0, 1), size_hint_y=None, height=50)
        self.messages.add_widget(message)

    def update_frame(self, frame):
        # Convert frame to texture
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def schedule_update(self, get_frame,ip,UI):
        Clock.schedule_interval(lambda dt: self.update_frame(get_frame(ip,UI)), 1.0 / 30.0)
        
    def promote_pawn(self):
        self.user_choice_event = Event()
        self.promotion_choice = None
        Clock.schedule_once(self._create_promotion_popup)
        self.user_choice_event.wait()  # Wait here until the user makes a choice
        return self.promotion_choice

    def _create_promotion_popup(self, dt):
        def on_button_click(instance):
            self.promotion_choice = instance.piece_id  # Retrieve custom attribute
            self.user_choice_event.set()  # Signal that the user made a choice
            popup.dismiss()

        content = BoxLayout(orientation='vertical', spacing=5)
        pieces = {"Queen": "q", "Rook": "r", "Bishop": "b", "Knight": "n"}
        for piece, symbol in pieces.items():
            btn = Button(text=piece, size_hint_y=None, height=40)
            btn.piece_id = symbol  # Set custom attribute
            btn.bind(on_release=on_button_click)
            content.add_widget(btn)

        popup = Popup(title='Pawn Promotion', content=content, size_hint=(None, None), size=(200, 200), auto_dismiss=False)
        popup.open()
        
    def select_difficulty(self):
        self.user_choice_event_2 = Event()
        self.difficulty_selection = None
        Clock.schedule_once(self._create_difficulty_popup)
        self.user_choice_event_2.wait()  # Wait here until the user makes a choice
        return self.difficulty_selection

    def _create_difficulty_popup(self, dt):
        def on_button_click(instance):
            self.difficulty_selection = instance.diff_id  # Retrieve custom attribute
            self.user_choice_event_2.set()  # Signal that the user made a choice
            popup.dismiss()
        content = BoxLayout(orientation='vertical', spacing=5)
        differences = {"Infant": 1, "Beginner": 2, "Intermediate": 3, "Grand Master": 4}
        for diff, id in differences.items():
            btn = Button(text=diff, size_hint_y=None, height=40)
            btn.diff_id = id  # Set custom attribute
            btn.bind(on_release=on_button_click)
            content.add_widget(btn)

        popup = Popup(title='Difficulty Selection', content=content, size_hint=(None, None), size=(200, 200), auto_dismiss=False)
        popup.open()
        
    def display_chessboard_image(self,pieces_setup):
        global symbols_folder
        # Assuming draw_chess_game returns a PIL Image
        pil_image = cviz.draw_chess_game(pieces_setup, symbols_folder)
        Clock.schedule_once(lambda dt:self._switch_image(pil_image))
        
    def _switch_image(self,to):
        # Save the PIL image to a temporary file
        temp_image_path = symbols_folder + '/temp.png'
        to.save(temp_image_path)
        
        self.board_viz.texture = None
        self.board_viz.texture = CoreImage(temp_image_path,nocache=True).texture
        
        # Optionally, delete the temporary file after loading
        os.remove(temp_image_path)
