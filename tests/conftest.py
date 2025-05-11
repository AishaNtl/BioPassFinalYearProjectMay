# tests/conftest.py
import pytest
from unittest import mock
import simpleaudio
import dlib
import cv2
import sys
from unittest.mock import MagicMock

@pytest.fixture(autouse=True, scope="session")
def disable_hardware():
    """Globally disable all hardware-dependent operations during testing"""
    
    # ===== Audio Mocking =====
    # Mock the simpleaudio module to prevent actual audio operations during tests
    simpleaudio.play = mock.MagicMock(return_value=mock.MagicMock())
    simpleaudio.WaveObject.from_wave_file = mock.MagicMock(return_value=mock.MagicMock())

    # ===== Vision Mocking =====
    # Mock dlib face detector
    mock_detector = mock.MagicMock()
    mock_detector.return_value = [mock.MagicMock()]  # Always returns at least one face
    
    # Mock dlib shape predictor
    mock_predictor = mock.MagicMock()
    
    # Mock OpenCV video capture
    mock_video = mock.MagicMock()
    mock_video.isOpened.return_value = True
    mock_video.read.return_value = (True, None)  # (ret, frame)
    
    # ===== Apply All Mocks =====
    with mock.patch('dlib.get_frontal_face_detector', mock_detector), \
         mock.patch('dlib.shape_predictor', mock_predictor), \
         mock.patch('cv2.VideoCapture', return_value=mock_video), \
         mock.patch('simpleaudio.WaveObject.from_wave_file', mock.MagicMock()), \
         mock.patch('simpleaudio.play', mock.MagicMock()):
        
        # Mock cv2.imshow globally during tests (suppress actual GUI popups)
        original_imshow = cv2.imshow
        cv2.imshow = mock.MagicMock()

        yield  # Tests run here
        
        # Restore original function after tests
        cv2.imshow = original_imshow

    # ===== Audio Module Override =====
    # Mocking sys.modules to disable simpleaudio completely during tests
    sys.modules['simpleaudio'] = MagicMock()

