import os
import runpy

# Streamlit App Entrypoint for Hugging Face Spaces & Streamlit Cloud
# This replaces Hugging Face's boilerplate template and redirects to our real app in scripts/app.py
if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "app.py")
    runpy.run_path(script_path, run_name="__main__")
