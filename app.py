import cv2
import pygame
import numpy as np
import time
import math
import arabic_reshaper
from bidi.algorithm import get_display
import mediapipe as mp
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# Initialize Pygame
pygame.init()
pygame.font.init()

# Constants (Dynamic - will be updated based on screen size)
SCREEN_WIDTH = 1400  # Default windowed size
SCREEN_HEIGHT = 900  # Default windowed size
FPS = 60

# Colors (Modern Design)
class Colors:
    BACKGROUND = (15, 15, 25)  # Dark blue-black
    SURFACE = (25, 30, 45)     # Dark surface
    PRIMARY = (100, 200, 255)  # Light blue
    SECONDARY = (80, 120, 200) # Blue
    ACCENT = (255, 180, 0)     # Orange
    SUCCESS = (50, 200, 100)   # Green
    WARNING = (255, 150, 50)   # Orange
    ERROR = (255, 80, 80)      # Red
    TEXT_PRIMARY = (255, 255, 255)    # White
    TEXT_SECONDARY = (180, 180, 200)  # Light gray
    KEY_DEFAULT = (40, 45, 60)        # Dark gray
    KEY_HOVER = (60, 70, 90)          # Lighter gray
    KEY_PRESS = (80, 100, 130)        # Even lighter
    KEY_SELECTED = (100, 200, 255)    # Blue highlight

@dataclass
class Key:
    char: str
    arabic_char: str
    rect: pygame.Rect
    is_special: bool = False
    is_pressed: bool = False
    is_hovered: bool = False
    press_start_time: float = 0
    press_progress: float = 0

class Layout(Enum):
    ENGLISH = "english"
    ARABIC = "arabic"

class VirtualKeyboard:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.current_layout = Layout.ENGLISH
        self.typed_text = ""
        self.press_duration = 1.0  # seconds
        
        # Fonts
        self.font_key = pygame.font.Font(None, 28)
        self.font_text = pygame.font.Font(None, 36)
        self.font_arabic = pygame.font.Font(None, 32)
        
        # Try to load system Arabic fonts
        arabic_fonts = [
            "arial.ttf",  # Windows
            "Arial Unicode MS.ttf",  # Mac
            "/System/Library/Fonts/Arial.ttf",  # Mac
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf",  # Windows full path
            "C:/Windows/Fonts/tahoma.ttf",  # Windows Tahoma
            "C:/Windows/Fonts/calibri.ttf",  # Windows Calibri
        ]
        
        self.arabic_font_loaded = False
        for font_path in arabic_fonts:
            try:
                self.font_arabic = pygame.font.Font(font_path, 32)
                self.font_key_arabic = pygame.font.Font(font_path, 28)
                print(f"Arabic font loaded: {font_path}")
                self.arabic_font_loaded = True
                
                # Test Arabic character rendering
                test_surface = self.font_key_arabic.render('ض', True, (255, 255, 255))
                if test_surface.get_width() == 0:
                    print(f"Warning: Font {font_path} doesn't render Arabic properly")
                    continue
                else:
                    print(f"Arabic font working properly: {font_path}")
                    break
            except Exception as e:
                print(f"Failed to load font {font_path}: {e}")
                continue
        else:
            # Fallback to default font with larger size for Arabic
            self.font_arabic = pygame.font.Font(None, 36)
            self.font_key_arabic = pygame.font.Font(None, 32)
            print("Using default font for Arabic - Arabic characters may not display properly")
            self.arabic_font_loaded = False
        
        # Initialize keys
        self.init_keys()
        
    def init_keys(self):
        """Initialize keyboard keys layout"""
        self.keys = []
        
        # Key layout
        english_rows = [
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACK', 'CLEAR']
        ]
        
        arabic_chars = {
            'Q': 'ض', 'W': 'ص', 'E': 'ث', 'R': 'ق', 'T': 'ف',
            'Y': 'غ', 'U': 'ع', 'I': 'ه', 'O': 'خ', 'P': 'ح',
            'A': 'ش', 'S': 'س', 'D': 'ي', 'F': 'ب', 'G': 'ل',
            'H': 'ا', 'J': 'ت', 'K': 'ن', 'L': 'م',
            'Z': 'ظ', 'X': 'ط', 'C': 'ذ', 'V': 'د', 'B': 'ز',
            'N': 'ر', 'M': 'و'
        }
        
        # Key dimensions
        key_width = 70
        key_height = 60
        key_margin = 8
        
        # Starting position (centered)
        start_x = (self.screen_width - (10 * (key_width + key_margin))) // 2
        start_y = self.screen_height - 280
        
        for row_idx, row in enumerate(english_rows):
            row_width = len(row) * (key_width + key_margin)
            if row_idx == 3:  # Special keys row
                row_width = 3 * (key_width * 2 + key_margin)
            
            row_start_x = (self.screen_width - row_width) // 2
            
            for key_idx, char in enumerate(row):
                x = row_start_x + key_idx * (key_width + key_margin)
                if row_idx == 3:  # Special keys
                    x = row_start_x + key_idx * (key_width * 2 + key_margin)
                    current_key_width = key_width * 2
                else:
                    current_key_width = key_width
                
                y = start_y + row_idx * (key_height + key_margin)
                
                arabic_char = arabic_chars.get(char, char)
                is_special = char in ['SPACE', 'BACK', 'CLEAR']
                
                key = Key(
                    char=char,
                    arabic_char=arabic_char,
                    rect=pygame.Rect(x, y, current_key_width, key_height),
                    is_special=is_special
                )
                self.keys.append(key)
    
    def switch_layout(self):
        """Switch between English and Arabic layouts"""
        if self.current_layout == Layout.ENGLISH:
            self.current_layout = Layout.ARABIC
        else:
            self.current_layout = Layout.ENGLISH
        print(f"Switched to {self.current_layout.value} layout")
    
    def get_display_char(self, key: Key) -> str:
        """Get the character to display on key based on current layout"""
        if key.char == 'BACK':
            return 'BACK'
        elif key.char == 'SPACE':
            return 'SPACE'
        elif key.char == 'CLEAR':
            return 'CLEAR'
        elif self.current_layout == Layout.ARABIC and not key.is_special:
            # For Arabic, use the Arabic character directly without reshaping
            # since we're only displaying single characters on keys
            return key.arabic_char
        else:
            return key.char
    
    def process_finger_position(self, finger_pos: Tuple[int, int]) -> Optional[Key]:
        """Process finger position and return selected key"""
        selected_key = None
        current_time = time.time()
        
        # Reset all hover states
        for key in self.keys:
            key.is_hovered = False
            if not key.is_pressed:
                key.press_progress = 0
        
        # Check which key is being pointed at
        for key in self.keys:
            if key.rect.collidepoint(finger_pos):
                key.is_hovered = True
                selected_key = key
                
                # Start press timer if not already pressing
                if not key.is_pressed:
                    key.is_pressed = True
                    key.press_start_time = current_time
                
                # Calculate press progress
                elapsed = current_time - key.press_start_time
                key.press_progress = min(elapsed / self.press_duration, 1.0)
                
                # Execute key press if held long enough
                if key.press_progress >= 1.0:
                    self.execute_key_press(key)
                    key.is_pressed = False
                    key.press_progress = 0
                
                break
        
        # Reset press state for keys not being pointed at
        for key in self.keys:
            if not key.is_hovered and key.is_pressed:
                key.is_pressed = False
                key.press_progress = 0
        
        return selected_key
    
    def execute_key_press(self, key: Key):
        """Execute the key press action"""
        if key.char == 'SPACE':
            self.typed_text += ' '
        elif key.char == 'BACK':
            self.typed_text = self.typed_text[:-1]
        elif key.char == 'CLEAR':
            self.typed_text = ""
        else:
            if self.current_layout == Layout.ARABIC:
                self.typed_text += key.arabic_char
            else:
                self.typed_text += key.char.lower()
        
        print(f"Key pressed: {key.char} -> Text: {self.typed_text}")
    
    def get_display_text(self) -> str:
        """Get properly formatted text for display"""
        if not self.typed_text:
            return ""
            
        if self.current_layout == Layout.ARABIC and self.typed_text:
            # For Arabic text, apply reshaping and BiDi
            try:
                # First check if the text contains Arabic characters
                has_arabic = any('\u0600' <= char <= '\u06FF' for char in self.typed_text)
                if has_arabic:
                    reshaped_text = arabic_reshaper.reshape(self.typed_text)
                    return get_display(reshaped_text)
                else:
                    return self.typed_text
            except Exception as e:
                print(f"Arabic text processing error: {e}")
                return self.typed_text
        return self.typed_text
    
    def draw(self, screen: pygame.Surface, selected_key: Optional[Key] = None):
        """Draw the virtual keyboard"""
        # Draw keys
        for key in self.keys:
            # Determine key color
            if key.is_pressed and key.press_progress > 0:
                # Gradient color during press
                progress = key.press_progress
                color = (
                    int(Colors.KEY_HOVER[0] + (Colors.SUCCESS[0] - Colors.KEY_HOVER[0]) * progress),
                    int(Colors.KEY_HOVER[1] + (Colors.SUCCESS[1] - Colors.KEY_HOVER[1]) * progress),
                    int(Colors.KEY_HOVER[2] + (Colors.SUCCESS[2] - Colors.KEY_HOVER[2]) * progress)
                )
            elif key.is_hovered:
                color = Colors.KEY_HOVER
            else:
                color = Colors.KEY_DEFAULT
            
            # Draw key background with rounded corners
            pygame.draw.rect(screen, color, key.rect, border_radius=8)
            pygame.draw.rect(screen, Colors.TEXT_SECONDARY, key.rect, 2, border_radius=8)
            
            # Draw key text
            display_char = self.get_display_char(key)
            
            # Use appropriate font for Arabic or English
            if self.current_layout == Layout.ARABIC and not key.is_special:
                # For Arabic characters on keys, try to render properly
                try:
                    # Test if character renders properly
                    test_surface = self.font_key_arabic.render(display_char, True, Colors.TEXT_PRIMARY)
                    if test_surface.get_width() > 0:  # Character rendered successfully
                        text_surface = test_surface
                    else:
                        # Fallback to default font
                        text_surface = self.font_key.render(display_char, True, Colors.TEXT_PRIMARY)
                except:
                    # Fallback to default font
                    text_surface = self.font_key.render(display_char, True, Colors.TEXT_PRIMARY)
            else:
                text_surface = self.font_key.render(display_char, True, Colors.TEXT_PRIMARY)
            
            text_rect = text_surface.get_rect(center=key.rect.center)
            screen.blit(text_surface, text_rect)
    
    def draw_text_display(self, screen: pygame.Surface):
        """Draw the text display area"""
        # Text display background
        text_rect = pygame.Rect(50, 50, self.screen_width - 100, 100)
        pygame.draw.rect(screen, Colors.SURFACE, text_rect, border_radius=10)
        pygame.draw.rect(screen, Colors.PRIMARY, text_rect, 3, border_radius=10)
        
        # Display text
        display_text = self.get_display_text()
        if not display_text:
            if self.current_layout == Layout.ARABIC:
                display_text = "ابدأ الكتابة بيديك..."
            else:
                display_text = "Start typing with your hands..."
        
        if self.current_layout == Layout.ARABIC:
            text_surface = self.font_arabic.render(display_text, True, Colors.TEXT_PRIMARY)
        else:
            text_surface = self.font_text.render(display_text, True, Colors.TEXT_PRIMARY)
        
        # Center text in display area
        text_rect_centered = text_surface.get_rect()
        text_rect_centered.center = text_rect.center
        screen.blit(text_surface, text_rect_centered)
        
        # Layout indicator
        layout_text = f"Layout: {self.current_layout.value.upper()}"
        layout_surface = self.font_key.render(layout_text, True, Colors.ACCENT)
        screen.blit(layout_surface, (60, 160))
        
        # Arabic font status indicator
        if hasattr(self, 'arabic_font_loaded'):
            font_status = "" if self.arabic_font_loaded else "Arabic Font: Default (Limited)"
            font_color = Colors.SUCCESS if self.arabic_font_loaded else Colors.WARNING
            font_surface = self.font_key.render(font_status, True, font_color)
            screen.blit(font_surface, (300, 160))

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hands(self, frame):
        """Detect hands and return hand information"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        detected_hands = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                # Convert landmarks to pixel coordinates
                h, w, c = frame.shape
                landmarks = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy])
                
                detected_hands.append({
                    'label': hand_label,
                    'landmarks': landmarks,
                    'raw_landmarks': hand_landmarks
                })
                
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return detected_hands
    
    def get_finger_tip(self, landmarks, finger='INDEX'):
        """Get finger tip coordinates"""
        finger_tips = {
            'THUMB': 4, 'INDEX': 8, 'MIDDLE': 12, 'RING': 16, 'PINKY': 20
        }
        
        if finger in finger_tips:
            return landmarks[finger_tips[finger]]
        return None
    
    def is_finger_up(self, landmarks, finger):
        """Check if finger is up"""
        finger_indices = {
            'THUMB': [4, 3], 'INDEX': [8, 6], 'MIDDLE': [12, 10],
            'RING': [16, 14], 'PINKY': [20, 18]
        }
        
        if finger not in finger_indices:
            return False
        
        tip_idx, joint_idx = finger_indices[finger]
        
        if finger == 'THUMB':
            return landmarks[tip_idx][0] > landmarks[joint_idx][0]
        else:
            return landmarks[tip_idx][1] < landmarks[joint_idx][1]
    
    def detect_victory_sign(self, landmarks):
        """Detect victory sign (peace sign)"""
        fingers_up = []
        for finger in ['THUMB', 'INDEX', 'MIDDLE', 'RING', 'PINKY']:
            fingers_up.append(self.is_finger_up(landmarks, finger))
        
        # Victory: index and middle up, others down
        return fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]

class ProgressBar:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.progress = 0.0
        
    def update(self, progress):
        self.progress = max(0.0, min(1.0, progress))
    
    def draw(self, screen):
        if self.progress > 0:
            # Background
            pygame.draw.rect(screen, Colors.SURFACE, self.rect, border_radius=15)
            pygame.draw.rect(screen, Colors.TEXT_SECONDARY, self.rect, 2, border_radius=15)
            
            # Progress fill
            fill_height = int(self.rect.height * self.progress)
            fill_rect = pygame.Rect(
                self.rect.x, 
                self.rect.y + self.rect.height - fill_height,
                self.rect.width, 
                fill_height
            )
            pygame.draw.rect(screen, Colors.SUCCESS, fill_rect, border_radius=15)
            
            # Text
            font = pygame.font.Font(None, 24)
            text = font.render("HOLD", True, Colors.TEXT_PRIMARY)
            text_rect = text.get_rect(center=(self.rect.centerx, self.rect.y - 20))
            screen.blit(text, text_rect)

class ModernVirtualKeyboard:
    def __init__(self):
        # Initialize pygame display
        self.is_fullscreen = True
        self.windowed_size = (1400, 900)
        
        # Start in fullscreen
        self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        SCREEN_WIDTH = self.screen.get_width()
        SCREEN_HEIGHT = self.screen.get_height()
        
        pygame.display.set_caption("Modern Virtual Keyboard with Hand Tracking")
        self.clock = pygame.time.Clock()
        
        # Initialize components with dynamic screen size
        self.keyboard = VirtualKeyboard(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.hand_detector = HandDetector()
        self.progress_bar = ProgressBar(SCREEN_WIDTH - 80, SCREEN_HEIGHT - 350, 40, 200)
        
        # Camera
        self.camera = None
        self.camera_active = False
        self.init_camera()
        
        # Victory sign detection
        self.victory_sign_start_time = None
        self.layout_switch_duration = 2.0
        
        # Camera display (dynamic positioning)
        self.update_camera_rect()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 48)
        
        print(f"Started in fullscreen mode: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print("Controls:")
        print("   F11 - Toggle fullscreen/windowed")
        print("   ESC - Quit application")
    
    def update_camera_rect(self):
        """Update camera rectangle position based on screen size"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        self.camera_rect = pygame.Rect(screen_width - 320, 20, 300, 200)
        
        # Update progress bar position
        self.progress_bar.rect.x = screen_width - 80
        self.progress_bar.rect.y = screen_height - 350
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if self.is_fullscreen:
            # Switch to windowed
            self.screen = pygame.display.set_mode(self.windowed_size)
            self.is_fullscreen = False
            print(f"Switched to windowed mode: {self.windowed_size[0]}x{self.windowed_size[1]}")
        else:
            # Switch to fullscreen
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.is_fullscreen = True
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            print(f"Switched to fullscreen mode: {screen_width}x{screen_height}")
        
        # Update all UI elements for new screen size
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Recreate keyboard with new dimensions
        self.keyboard = VirtualKeyboard(screen_width, screen_height)
        self.update_camera_rect()
    
    def draw_fullscreen_indicator(self):
        """Draw fullscreen/windowed indicator"""
        mode_text = "FULLSCREEN" if self.is_fullscreen else "WINDOWED"
        mode_color = Colors.SUCCESS if self.is_fullscreen else Colors.WARNING
        
        mode_surface = self.font_ui.render(f"{mode_text} (F11 to toggle)", True, mode_color)
        
        # Position in top left
        self.screen.blit(mode_surface, (20, 20))
    
    def draw_enhanced_ui(self, detected_hands, victory_signs):
        """Enhanced UI with fullscreen support"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        # Draw fullscreen indicator
        self.draw_fullscreen_indicator()
        
        # Title (centered better)
        title_surface = self.font_title.render(" ", True, Colors.PRIMARY)
        title_rect = title_surface.get_rect()
        title_x = 50
        title_y = max(70, screen_height * 0.08)  # Responsive positioning
        self.screen.blit(title_surface, (title_x, title_y))
        
        # Instructions (responsive positioning)
        instructions = []
        
        instructions_y = title_y + 60
        for i, instruction in enumerate(instructions):
            text_surface = self.font_ui.render(instruction, True, Colors.TEXT_SECONDARY)
            self.screen.blit(text_surface, (50, instructions_y + i * 25))
        
        # Hands detected counter
        hands_y = instructions_y + len(instructions) * 25 + 20
        hands_text = f""
        hands_surface = self.font_ui.render(hands_text, True, Colors.ACCENT)
        self.screen.blit(hands_surface, (50, hands_y))
        
        # Victory sign progress
        if victory_signs >= 2:
            if self.victory_sign_start_time:
                elapsed = time.time() - self.victory_sign_start_time
                progress = min(elapsed / self.layout_switch_duration, 1.0)
                
                # Progress bar for layout switch
                bar_y = hands_y + 40
                bar_rect = pygame.Rect(50, bar_y + 20, 300, 20)
                pygame.draw.rect(self.screen, Colors.SURFACE, bar_rect, border_radius=10)
                
                fill_width = int(bar_rect.width * progress)
                fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
                pygame.draw.rect(self.screen, Colors.WARNING, fill_rect, border_radius=10)
                
                # Text
                progress_text = f"Hold victory sign to switch layout... {int(progress * 100)}%"
                progress_surface = self.font_ui.render(progress_text, True, Colors.WARNING)
                self.screen.blit(progress_surface, (50, bar_y))
        
        # Draw performance info (FPS)
        fps = int(self.clock.get_fps())
        fps_color = Colors.SUCCESS if fps > 50 else Colors.WARNING if fps > 30 else Colors.ERROR
        fps_text = f"FPS: {fps}"
        fps_surface = self.font_ui.render(fps_text, True, fps_color)
        self.screen.blit(fps_surface, (screen_width - 150, 20))
    
    def draw_exit_hint(self):
        """Draw exit hint in corner"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        if self.is_fullscreen:
            exit_text = "Press ESC to exit fullscreen | F11 to toggle"
            exit_surface = self.font_ui.render(exit_text, True, Colors.TEXT_SECONDARY)
            exit_rect = exit_surface.get_rect()
            exit_rect.bottomright = (screen_width - 20, screen_height - 20)
            self.screen.blit(exit_surface, exit_rect)
        else:
            # In windowed mode, show different hint
            exit_text = "ESC to quit | F11 for fullscreen"
            exit_surface = self.font_ui.render(exit_text, True, Colors.TEXT_SECONDARY)
            exit_rect = exit_surface.get_rect()
            exit_rect.bottomright = (screen_width - 20, screen_height - 20)
            self.screen.blit(exit_surface, exit_rect)
    
    def draw_exit_hint(self):
        """Draw exit hint in corner"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        
        if self.is_fullscreen:
            exit_text = "Press ESC to exit fullscreen"
            exit_surface = self.font_ui.render(exit_text, True, Colors.TEXT_SECONDARY)
            exit_rect = exit_surface.get_rect()
            exit_rect.bottomright = (screen_width - 20, screen_height - 20)
            self.screen.blit(exit_surface, exit_rect)
    
    def init_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_active = True
                print("Camera initialized successfully")
            else:
                print("Failed to initialize camera")
        except Exception as e:
            print(f"Camera error: {e}")
    
    def draw_gradient_background(self):
        """Draw a beautiful gradient background (responsive)"""
        screen_height = self.screen.get_height()
        screen_width = self.screen.get_width()
        
        for y in range(screen_height):
            ratio = y / screen_height
            r = int(Colors.BACKGROUND[0] + (Colors.SURFACE[0] - Colors.BACKGROUND[0]) * ratio)
            g = int(Colors.BACKGROUND[1] + (Colors.SURFACE[1] - Colors.BACKGROUND[1]) * ratio)
            b = int(Colors.BACKGROUND[2] + (Colors.SURFACE[2] - Colors.BACKGROUND[2]) * ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (screen_width, y))
    
    def draw_camera_feed(self, detected_hands, finger_tip):
        """Draw camera feed in top right corner (responsive)"""
        if self.camera and self.camera_active:
            ret, frame = self.camera.read()
            if ret:
                # Flip frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Draw finger tip circle
                if finger_tip:
                    # Convert screen coordinates back to camera coordinates
                    screen_width = self.screen.get_width()
                    screen_height = self.screen.get_height()
                    cam_x = int((finger_tip[0] / screen_width) * 640)
                    cam_y = int((finger_tip[1] / screen_height) * 480)
                    cv2.circle(frame, (cam_x, cam_y), 15, (0, 255, 0), -1)
                
                # Draw hand count
                cv2.putText(frame, f"Hands: {len(detected_hands)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw mode indicator
                mode_text = "FULLSCREEN" if self.is_fullscreen else "WINDOWED"
                cv2.putText(frame, mode_text, (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw hand labels
                for hand_info in detected_hands:
                    landmarks = hand_info['landmarks']
                    label = hand_info['label']
                    wrist = landmarks[0]
                    cv2.putText(frame, f"{label} Hand", (wrist[0], wrist[1] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Convert to pygame surface
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                frame_surface = pygame.transform.scale(frame_surface, (self.camera_rect.width, self.camera_rect.height))
                
                # Draw camera background with rounded corners
                pygame.draw.rect(self.screen, Colors.SURFACE, self.camera_rect, border_radius=15)
                pygame.draw.rect(self.screen, Colors.PRIMARY, self.camera_rect, 3, border_radius=15)
                
                # Draw frame
                self.screen.blit(frame_surface, self.camera_rect)
                
                # Camera label
                cam_label = "Live Camera"
                cam_surface = self.font_ui.render(cam_label, True, Colors.PRIMARY)
                label_rect = cam_surface.get_rect()
                label_rect.centerx = self.camera_rect.centerx
                label_rect.bottom = self.camera_rect.top - 5
                self.screen.blit(cam_surface, label_rect)
    
    def draw_ui_elements(self, detected_hands, victory_signs):
        """Legacy method - replaced by draw_enhanced_ui"""
        # This method is kept for compatibility but redirects to enhanced version
        self.draw_enhanced_ui(detected_hands, victory_signs)
    
    def run(self):
        """Main game loop with fullscreen support"""
        running = True
        
        print("Application started!")
        print("Keyboard shortcuts:")
        print("   F11 - Toggle fullscreen/windowed mode")
        print("   ESC - Exit application")
        print("   Use index finger to interact with keyboard")
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.is_fullscreen:
                            # First ESC in fullscreen toggles to windowed
                            self.toggle_fullscreen()
                        else:
                            # ESC in windowed mode quits
                            running = False
                    elif event.key == pygame.K_F11:
                        # F11 toggles fullscreen mode
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_F1:
                        # F1 shows help (optional)
                        print("🆘 Help:")
                        print("   F11 - Toggle fullscreen")
                        print("   ESC - Exit (or exit fullscreen first)")
                        print("   Point index finger to select keys")
                        print("   Hold for 1 second to press key")
                        print("   Victory sign with both hands = switch layout")
            
            # Clear screen with gradient
            self.draw_gradient_background()
            
            # Camera and hand detection
            detected_hands = []
            finger_tip = None
            victory_signs = 0
            
            if self.camera and self.camera_active:
                ret, frame = self.camera.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    detected_hands = self.hand_detector.detect_hands(frame)
                    
                    # Process hands
                    for hand_info in detected_hands:
                        landmarks = hand_info['landmarks']
                        
                        # Get finger tip for first hand
                        if finger_tip is None:
                            tip = self.hand_detector.get_finger_tip(landmarks, 'INDEX')
                            if tip:
                                # Convert camera coordinates to screen coordinates
                                screen_width = self.screen.get_width()
                                screen_height = self.screen.get_height()
                                screen_x = int((tip[0] / 640) * screen_width)
                                screen_y = int((tip[1] / 480) * screen_height)
                                finger_tip = (screen_x, screen_y)
                        
                        # Check victory sign
                        if self.hand_detector.detect_victory_sign(landmarks):
                            victory_signs += 1
            
            # Handle layout switching
            if victory_signs >= 2:
                if self.victory_sign_start_time is None:
                    self.victory_sign_start_time = time.time()
                
                elapsed = time.time() - self.victory_sign_start_time
                if elapsed >= self.layout_switch_duration:
                    self.keyboard.switch_layout()
                    self.victory_sign_start_time = None
            else:
                self.victory_sign_start_time = None
            
            # Process keyboard interaction
            selected_key = None
            if finger_tip:
                selected_key = self.keyboard.process_finger_position(finger_tip)
                
                # Draw finger tip cursor with glow effect
                cursor_color = Colors.SUCCESS if selected_key else Colors.PRIMARY
                
                # Glow effect
                for radius in range(25, 10, -3):
                    alpha = int(50 * (25 - radius) / 15)
                    glow_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, (*cursor_color, alpha), (radius, radius), radius)
                    glow_rect = glow_surface.get_rect(center=finger_tip)
                    self.screen.blit(glow_surface, glow_rect)
                
                # Main cursor
                pygame.draw.circle(self.screen, cursor_color, finger_tip, 12)
                pygame.draw.circle(self.screen, Colors.TEXT_PRIMARY, finger_tip, 12, 2)
                
                # Cursor crosshair
                pygame.draw.line(self.screen, Colors.TEXT_PRIMARY, 
                               (finger_tip[0] - 8, finger_tip[1]), 
                               (finger_tip[0] + 8, finger_tip[1]), 2)
                pygame.draw.line(self.screen, Colors.TEXT_PRIMARY, 
                               (finger_tip[0], finger_tip[1] - 8), 
                               (finger_tip[0], finger_tip[1] + 8), 2)
            
            # Update progress bar
            if selected_key and selected_key.is_pressed:
                self.progress_bar.update(selected_key.press_progress)
            else:
                self.progress_bar.update(0)
            
            # Draw all components
            self.keyboard.draw_text_display(self.screen)
            self.keyboard.draw(self.screen, selected_key)
            self.progress_bar.draw(self.screen)
            self.draw_enhanced_ui(detected_hands, victory_signs)
            self.draw_camera_feed(detected_hands, finger_tip)
            self.draw_exit_hint()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Cleanup
        print("Shutting down...")
        if self.camera:
            self.camera.release()
        pygame.quit()
        sys.exit()

def main():
    
    try:
        app = ModernVirtualKeyboard()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Troubleshooting:")
        print("   • Make sure camera is not used by another app")
        print("   • Install required packages: pygame, opencv-python, mediapipe")
        print("   • Check if arabic-reshaper and python-bidi are installed")
    finally:
        try:
            pygame.quit()
        except:
            pass
        print("Goodbye!")
        sys.exit()

if __name__ == "__main__":
    main()