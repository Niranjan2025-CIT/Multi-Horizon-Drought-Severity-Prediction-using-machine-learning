"""
DROUGHT-X System Integrity Test
Verifies that all production models, backend dependencies, and critical data assets
are correctly located and functional.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestDroughtXIntegrity(unittest.TestCase):

    def test_production_models_exist(self):
        """Verify all 4 horizon model files exist in models/"""
        for week in range(1, 5):
            model_path = os.path.join(BASE_DIR, "models", f"model_{week}w.pkl")
            self.assertTrue(
                os.path.exists(model_path),
                f"Production model missing: models/model_{week}w.pkl"
            )

    def test_production_data_files_exist(self):
        """Verify all essential data files are present"""
        expected_files = [
            os.path.join(BASE_DIR, "data", "soil_data.csv"),
            os.path.join(BASE_DIR, "data", "texas_county_locations.csv"),
            os.path.join(BASE_DIR, "data", "county_neighbors.json"),
            os.path.join(BASE_DIR, "data", "texas_geojson.json"),
            os.path.join(BASE_DIR, "data", "processed", "processed_weekly.csv"),
        ]
        for f in expected_files:
            self.assertTrue(os.path.exists(f), f"Required data file missing: {f}")

    def test_backend_imports(self):
        """Verify all backend modules can be imported cleanly"""
        import backend.main
        import backend.prediction_pipeline
        import backend.weather
        import backend.usdm
        import backend.neighbours
        import backend.historical

        self.assertIsNotNone(backend.main.app)
        self.assertGreater(len(backend.neighbours.neighbors), 0)


if __name__ == "__main__":
    unittest.main()
