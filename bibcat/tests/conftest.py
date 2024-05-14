

import os
import tempfile


def pytest_sessionstart(session):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Set the environment variable
    os.environ['BIBCAT_OUTPUT_DIR'] = temp_dir