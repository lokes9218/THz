"""
Unit tests for core engine modules.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.engine import SystemEngine
from src.core.power_manager import PowerManager

class TestCoreEngine(unittest.TestCase):
    
    def test_engine_config_load(self):
        engine = SystemEngine()
        self.assertIsNotNone(engine.config)
        self.assertEqual(engine.seed, 42)

    def test_power_manager_allocation(self):
        pm = PowerManager(max_power_watts=10.0)
        alloc = pm.allocate_power('holography_3d')
        self.assertEqual(alloc['allocated_watts'], 4.0)
        self.assertEqual(alloc['status'], 'NORMAL')

if __name__ == '__main__':
    unittest.main()
