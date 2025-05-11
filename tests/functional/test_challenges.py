import pytest
from biopass.add_faces3 import detect_expression_with_timer, expressions
from biopass.add_faces3 import (  # Import the neutral values directly
    neutral_eyebrow_height,
    neutral_mouth_height,
    neutral_mouth_width,
    neutral_ear
)

@pytest.fixture
def mock_calibration():
    """Mock calibration values for expression testing"""
    # Declare we're modifying the global variables
    global neutral_eyebrow_height, neutral_mouth_height, neutral_mouth_width, neutral_ear
    
    # Store original values
    original_values = {
        'eyebrow': neutral_eyebrow_height,
        'mouth_h': neutral_mouth_height,
        'mouth_w': neutral_mouth_width,
        'ear': neutral_ear
    }
    
    # Set mock values
    neutral_eyebrow_height = 15.0
    neutral_mouth_height = 20.0
    neutral_mouth_width = 50.0
    neutral_ear = 0.25
    
    yield  # Test runs here
    
    # Restore original values
    neutral_eyebrow_height = original_values['eyebrow']
    neutral_mouth_height = original_values['mouth_h']
    neutral_mouth_width = original_values['mouth_w']
    neutral_ear = original_values['ear']

class TestExpressionChallenges:
    #@pytest.mark.parametrize("expression", expressions.keys())
    @pytest.mark.parametrize("expression", expressions)
    def test_expression_detection(self, mock_calibration, expression):
        """Test expression detection with mock calibration"""
        result = detect_expression_with_timer(
            expression, 
            time_limit=3  # Reduced timeout for testing
        )
        assert isinstance(result, bool), "Should return boolean result"
        if result:
            print(f"Successfully detected {expression}")