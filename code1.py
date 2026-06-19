Python 3.12.1 (v3.12.1:2305ca5144, Dec  7 2023, 17:23:39) [Clang 13.0.0 (clang-1300.0.29.30)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np

def calculate_strides_and_fps(phase_frames, max_phase_frames, base_fps=25):
...     """
...     Calculates the adaptive stride and corresponding frames per second (FPS) 
...     for each surgical phase based on Formula (1) from the paper.
...     """
...     adaptive_results = {}
...     
...     for phase_id, p_frames in phase_frames.items():
...         # Formula 1: stride = floor((P / M) * 25)
...         stride = int(np.floor((p_frames / max_phase_frames) * base_fps))
...         # Ensure stride is at least 1 to prevent division by zero or infinite loops
...         stride = max(1, stride) 
...         
...         # Calculate the resulting adaptive sampling rate (FPS)
...         adaptive_fps = base_fps / stride
...         
...         adaptive_results[phase_id] = {
...             "stride": stride,
...             "adaptive_fps": round(adaptive_fps, 2)
...         }
...     return adaptive_results
... 
... # Total frame counts for each phase extracted from Table 1 of the paper
... phase_frames_data = {
...     "Phase 1": 214300,
...     "Phase 2": 1842525,  # Phase 2 has the highest number of frames (M)
...     "Phase 3": 352000,
...     "Phase 4": 1460825,
...     "Phase 5": 190450,
...     "Phase 6": 485650,
...     "Phase 7": 165925
... }
... 
... max_frames = phase_frames_data["Phase 2"]
... results = calculate_strides_and_fps(phase_frames_data, max_frames)
... 
... print("--- Adaptive Undersampling Strategy Results ---")
... for phase, info in results.items():
