# Jumbled Frames Reconstruction Challenge

This repository contains a solution to the **Jumbled Frames Reconstruction Challenge**.  
The solution reconstructs a single-shot 5s, 60 FPS video whose 300 frames have been randomly shuffled.

## Contents
- `reconstruct.py` — main script to reconstruct a jumbled video
- `requirements.txt` — Python dependencies
- `run_log.txt` — produced after running (execution timings)
- `frames/` — extracted frames (created at runtime)
- `reconstructed_fixed.mp4` — default output filename after running

---

## Requirements
- Python 3.8 or newer
- OS: Windows / Linux / macOS (tested on Windows)
- Recommended CPU: multi-core (script uses multiprocessing)

## Python dependencies
Install with pip:
```bash
pip install -r requirements.txt
