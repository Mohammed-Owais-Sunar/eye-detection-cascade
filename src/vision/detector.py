import cv2

class EyeDetector:
    def __init__(self):
        # Load pre-trained Haar Cascades directly from OpenCV's library
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # CLAHE for dynamic lighting adjustments in hallways
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
    def process_frame(self, frame):
        """Processes a frame, applies CLAHE, and tracks bounding boxes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        engagements = 0
        
        for (x, y, w, h) in faces:
            # Draw blue box for impressions (faces)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) > 0:
                engagements += 1
                for (ex, ey, ew, eh) in eyes:
                    # Draw green box for active engagement (eyes)
                    cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                    
        return frame, len(faces), engagements