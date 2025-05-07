# mmt

## Setting up a Python Virtual Environment

To create and activate a Python virtual environment, follow these steps:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install required dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

4. To deactivate the virtual environment:
   ```bash
   deactivate
   ```