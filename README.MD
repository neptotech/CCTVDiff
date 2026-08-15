# CCTVDiff
Detect and visualize motion in CCTV footage with movement detection with visualization, interactive heatmaps, and more.

---

This is a very powerful tool that comes with many cctv cameras. This project is mainly used for two major purpose:
1. Detect movement areas and level from the bottom bar graph
2. Heatmap showing were changes are happening with threshold.

**Flashback:** I developed it for fun in order to catch the hiding place of a rat in my house(This actually helped and then we put glue pads, adhesives etc to cover up/catch it). The software itself is meant to be used on stable colour camera footage with enough lighting because we arent using IR camera(I took videos by having tubelights on in kitchen the whole night).

### Download 
See releases for exe

---

## Features

### 🎥 Video Playback
- Open and play video files with intuitive controls
- Play/Pause and Stop functionality
- Adjustable playback speed (0.5x - 2.0x)
- Real-time playback mode toggle
- Frame-by-frame navigation
- Dynamic time display (current / total duration)

### 📊 Motion Detection & Analysis
- **Real-time motion scoring** - Analyzes every frame for movement
- **Interactive motion graph** - Visual representation of motion intensity over time
- **Graph zoom** - Scroll to zoom in/out, click to jump to specific timestamps
- **Multi-threaded analysis** - Runs detection in background without blocking playback
- Analysis step: 6 frames (configurable)
- GPU-accelerated computation when available

### 🌡️ Heatmap Visualization
- Visual heatmap overlay showing where motion is occurring
- **Adjustable opacity** (0-100%)
- **Configurable threshold** - Control sensitivity of motion detection
- **Custom color picker** - Choose your own heatmap color
- Real-time heatmap updates during playback

### ⌨️ Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Space` | Play / Pause |
| `← / →` | Seek ±10 seconds |
| `Shift+← / →` | Seek ±3 seconds |
| `Ctrl+← / →` | Seek ±1 minute |
| `Click seekbar` | Jump to position (respects zoom window) |
| `Click graph` | Jump to timestamp on graph |
| `Scroll graph` | Zoom in / out around cursor |

### 💾 Data Management
- **Export motion data** - Save motion intensity scores to JSON
- **Import motion data** - Load previously analyzed motion scores
- Persist analysis results for later review

### 🚀 GPU Acceleration
Automatic acceleration detection for video analysis:
- **CUDA** - NVIDIA GPU support
- **OpenCL** - iGPU, AMD, Intel support (default fallback)
- **CPU** - Full CPU fallback support

### 🎨 UI Customization
- Dark theme optimized interface
- Customizable heatmap colors
- Adjustable opacity levels for overlay
- Motion sensitivity threshold adjustment
- Responsive layout

### ⚙️ Installation
```bash
pip install requirements.txt
```

### 📝 Dependencies
- OpenCV (cv2)
- Pillow (PIL)
- NumPy
- tkinter (included with Python)

