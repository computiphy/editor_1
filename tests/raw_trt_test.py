import sys
import os
sys.path.append(os.getcwd())
from src.segmentation.background_remover import BackgroundRemover

remover = BackgroundRemover(model="u2net", device="tensorrt")
remover._get_session()
print("Success")
