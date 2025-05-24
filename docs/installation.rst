Installation Guide
==================

Requirements
------------

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space

Quick Install
-------------

Install the Digital Twin T1D SDK using pip:

.. code-block:: bash

   pip install digitaltwin-t1d

Development Installation
------------------------

For development or to get the latest features:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/digitaltwin-t1d/sdk.git
   cd sdk
   
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install in development mode
   pip install -e .
   
   # Install development dependencies
   pip install -r requirements-dev.txt

Docker Installation
-------------------

For containerized deployment:

.. code-block:: bash

   docker pull digitaltwin/t1d-sdk:latest
   docker run -p 8080:8080 digitaltwin/t1d-sdk:latest

Verify Installation
-------------------

.. code-block:: python

   import sdk
   print(sdk.__version__)
   
   # Test basic functionality
   from sdk import DigitalTwinSDK
   sdk = DigitalTwinSDK(mode='demo')
   print("✅ SDK installed successfully!")

Platform-Specific Notes
-----------------------

**Windows**
   - Ensure Visual C++ Build Tools are installed for some dependencies
   - Use PowerShell or Command Prompt as Administrator

**macOS**
   - Xcode Command Line Tools may be required
   - Use ``brew install python@3.8`` if needed

**Linux**
   - Install python3-dev package: ``sudo apt-get install python3-dev``
   - May need to install additional system libraries

Troubleshooting
---------------

**Import Error**
   Make sure the virtual environment is activated and all dependencies are installed.

**Permission Denied**
   Use ``pip install --user`` or run with appropriate permissions.

**Memory Error**
   The SDK requires at least 4GB RAM. Close other applications if needed.

Next Steps
----------

- Check out the :doc:`quickstart` guide
- Explore :doc:`examples`
- Read about :doc:`api/sdk` 