import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock
from biopass.add_faces3 import calculate_ear

# Test image paths
TEST_IMAGES = {
    "neutral": "tests/test_images/neutral.jpg",
    "happy": "tests/test_images/happy.jpg",
    "angry": "tests/test_images/angry.jpg",
    "surprised": "tests/test_images/surprised.jpg"
}

@pytest.fixture
def mock_landmark():
    """Create mock facial landmarks"""
    def create_mock(values):
        return MagicMock(part=lambda i: MagicMock(x=values[i][0], y=values[i][1]))
    return create_mock

def test_face_detection(neutral_face):
    """Test face detection with mock detector"""
    with patch('biopass.add_faces3.detector') as mock_detector:
        mock_detector.return_value = [MagicMock()]  # Return one mock face
        from biopass.add_faces3 import detector
        faces = detector(cv2.cvtColor(neutral_face, cv2.COLOR_BGR2GRAY))
        assert len(faces) == 1, "Should detect exactly one face"

@pytest.mark.parametrize("emotion,expected", [
    ("happy", "Happy"),
    ("angry", "Frown"), 
    ("surprised", "Shocked")
])
def test_emotion_classification(emotion, expected, mock_landmark):
    """Test emotion classification with mock landmarks"""
    with patch('biopass.add_faces3.detect_expression') as mock_detect:
        mock_detect.return_value = expected
        from biopass.add_faces3 import detect_expression
        
        # Create mock landmarks for the emotion
        landmarks = mock_landmark({
            36: (0, 0),  # Left eye
            48: (0, 0),  # Mouth corner
            # Add other required landmarks
        })
        
        result = detect_expression(landmarks)
        assert result == expected, f"Incorrect emotion detection: {result}"