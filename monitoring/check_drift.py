#!/usr/bin/env python3
"""
Check drift for CI/CD pipeline.
Wrapper script for GitHub Actions.
"""

import sys
sys.path.insert(0, '/app')

from monitoring.drift_detection import check_drift

if __name__ == "__main__":
    drift_detected = check_drift()
    sys.exit(1 if drift_detected else 0)
