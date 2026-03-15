import sys
import os
# Ensure project root is in path
sys.path.append(os.getcwd())

from src.segmentation.background_remover import BackgroundRemover
import onnxruntime as ort

print("--- START DIAGNOSTIC ---")
try:
    remover = BackgroundRemover(model="u2net", device="tensorrt")
    print("BackgroundRemover initialized. Initializing session...")
    session = remover._get_session()
    print("SUCCESS! Providers:", session.get_providers())
except Exception as e:
    import traceback
    print("\n--- CRASH DETECTED ---")
    print(f"Exception Type: {type(e)}")
    print(f"Exception Message: {e}")
    traceback.print_exc()
    print("Available Providers:", ort.get_available_providers())
print("--- END DIAGNOSTIC ---")
