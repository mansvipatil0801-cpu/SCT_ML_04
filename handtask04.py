# Hand Gesture Recognition - Production Ready Version
import sys
import subprocess
import time

# Auto-install function
def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--quiet'])
        return True
    except:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name, '--user', '--quiet'])
            return True
        except:
            return False

# Import with auto-installation
def safe_import():
    """Safely import all required packages"""
    packages = {'cv2': 'opencv-python', 'mediapipe': 'mediapipe', 'numpy': 'numpy'}
    
    for module, package in packages.items():
        try:
            globals()[module] = __import__(module)
            if module == 'mediapipe':
                globals()['mp'] = globals()[module]
            elif module == 'numpy':
                globals()['np'] = globals()[module]
        except ImportError:
            print(f"Installing {package}...")
            if install_package(package):
                globals()[module] = __import__(module)
                if module == 'mediapipe':
                    globals()['mp'] = globals()[module]
                elif module == 'numpy':
                    globals()['np'] = globals()[module]
                print(f"✅ {package} installed successfully")
            else:
                print(f"❌ Failed to install {package}")
                return False
    return True

# Initialize imports
if not safe_import():
    print("❌ Package installation failed. Please run manually:")
    print("pip install opencv-python mediapipe numpy")
    sys.exit(1)

class HandGestureRecognizer:
    def __init__(self):
        """Initialize MediaPipe hands solution"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Hand landmark indices
        self.tip_ids = [4, 8, 12, 16, 20]  # Fingertips
        self.pip_ids = [3, 6, 10, 14, 18]  # PIP joints
        
        # Gesture database
        self.gestures = {
            (0, 0, 0, 0, 0): ("FIST", "✊", (0, 0, 255)),
            (1, 1, 1, 1, 1): ("OPEN", "✋", (0, 255, 0)),
            (0, 1, 1, 0, 0): ("PEACE", "✌️", (255, 0, 0)),
            (1, 0, 0, 0, 0): ("THUMB", "👍", (0, 255, 255)),
            (0, 1, 0, 0, 0): ("ONE", "☝️", (255, 255, 0)),
            (0, 1, 1, 0, 0): ("TWO", "✌️", (255, 0, 255)),
            (0, 1, 1, 1, 0): ("THREE", "🤟", (128, 255, 0)),
            (0, 1, 1, 1, 1): ("FOUR", "🖐️", (255, 128, 0)),
            (1, 1, 0, 0, 0): ("GUN", "🔫", (128, 0, 255)),
            (0, 0, 0, 0, 1): ("PINKY", "🤙", (255, 0, 128))
        }
        
    def count_fingers(self, landmarks):
        """Advanced finger counting with improved accuracy"""
        fingers = []
        
        # Convert landmarks to coordinate list
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x, lm.y])
        
        # Thumb (horizontal comparison)
        if points[self.tip_ids[0]][0] > points[self.tip_ids[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers (vertical comparison with improved logic)
        for i in range(1, 5):
            tip_y = points[self.tip_ids[i]][1]
            pip_y = points[self.pip_ids[i]][1]
            mcp_y = points[self.tip_ids[i] - 3][1]  # MCP joint
            
            # More robust finger detection
            if tip_y < pip_y and tip_y < mcp_y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return tuple(fingers)
    
    def get_gesture_info(self, finger_pattern):
        """Get gesture name, emoji, and color from finger pattern"""
        if finger_pattern in self.gestures:
            return self.gestures[finger_pattern]
        else:
            count = sum(finger_pattern)
            return (f"{count} FINGERS", f"{count}️⃣", (128, 128, 128))
    
    def draw_ui(self, img, gesture_name, emoji, color, finger_pattern, fps):
        """Draw user interface elements"""
        h, w = img.shape[:2]
        
        # Header background
        cv2.rectangle(img, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (0, 0), (w, 100), color, 3)
        
        # Title
        cv2.putText(img, "HAND GESTURE RECOGNITION", (10, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # FPS and instructions
        cv2.putText(img, f"FPS: {fps}", (w-120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Press 'Q' to quit", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Results background
        cv2.rectangle(img, (0, h-120), (w, h), (0, 0, 0), -1)
        cv2.rectangle(img, (0, h-120), (w, h), color, 3)
        
        # Gesture results
        cv2.putText(img, f"GESTURE: {gesture_name}", (10, h-85), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        cv2.putText(img, f"EMOJI: {emoji}", (10, h-50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
        
        # Finger pattern
        pattern_str = "".join(map(str, finger_pattern))
        cv2.putText(img, f"PATTERN: {pattern_str}", (10, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Gesture count
        cv2.putText(img, f"TOTAL: {sum(finger_pattern)} fingers", (w-200, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🖐️  HAND GESTURE RECOGNITION SYSTEM STARTING")
        print("="*60)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            # Try alternative camera indices
            for i in range(1, 4):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
        
        if not cap.isOpened():
            print("❌ Error: No camera found!")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
        print("\n📋 SUPPORTED GESTURES:")
        for pattern, (name, emoji, _) in self.gestures.items():
            print(f"   {name}: {emoji}")
        print("\n🎥 Starting recognition... Press 'Q' to quit\n")
        
        # Performance tracking
        fps_counter = 0
        fps_time = time.time()
        current_fps = 0
        
        while True:
            success, img = cap.read()
            if not success:
                print("❌ Failed to capture frame")
                break
            
            # Flip image horizontally for natural interaction
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_img)
            
            # Default values
            gesture_name, emoji, color = "NO HAND", "👋", (100, 100, 100)
            finger_pattern = (0, 0, 0, 0, 0)
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand skeleton with improved styling
                    self.mp_drawing.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Recognize gesture
                    finger_pattern = self.count_fingers(hand_landmarks)
                    gesture_name, emoji, color = self.get_gesture_info(finger_pattern)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Draw UI
            self.draw_ui(img, gesture_name, emoji, color, finger_pattern, current_fps)
            
            # Display frame
            cv2.imshow("Hand Gesture Recognition", img)
            
            # Handle key input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Application closed successfully")

def main():
    """Main function"""
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure your camera is connected and accessible")

if __name__ == "__main__":
    main()