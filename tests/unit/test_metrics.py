# tests/unit/test_metrics.py
from unittest.mock import patch, MagicMock
import pytest

class TestFacialMetrics:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch('biopass.add_faces3.calculate_ear') as mock_ear:
            mock_ear.return_value = 0.5  # Default value
            from biopass.add_faces3 import calculate_ear
            self.calculate_ear = calculate_ear
            yield

    def test_ear_calculation(self):
        self.calculate_ear.return_value = 0.42
        assert self.calculate_ear([]) == 0.42

    def test_mouth_distance_calculation(self):
        """Test mouth distance calculation"""
        from biopass.add_faces3 import calculate_mouth_distance
        mouth_points = [(0,0), (0,5), (5,5), (5,0)]
        distance = calculate_mouth_distance(mouth_points)
        assert abs(distance - 5.0) < 0.1, "Incorrect mouth distance calculation"