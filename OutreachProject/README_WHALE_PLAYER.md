# Bowhead Whale Call Interactive Player

## Overview

An interactive, browser-based Theremin-style synthesizer for exploring and playing back bowhead whale acoustic calls from the Beaufort Sea dataset.

## Features

- **7 Call Types**: Simple upsweeps, downsweeps, constant tones, complex calls, pulse trains, harmonic calls, and variable calls
- **Interactive Playback**: Uses Web Audio API to synthesize whale calls from actual frequency contours extracted from spectrograms
- **Real-time Visualization**: Display spectrograms and frequency contours with animated playback
- **Theremin-Style Synthesis**: Continuous frequency modulation that accurately represents whale vocalizations
- **70,000+ Classified Calls**: Drawn from manually classified dataset

## Why Theremin Instead of Piano?

Bowhead whale calls are **frequency-modulated (FM) sweeps** with continuous pitch changes, not discrete notes. A Theremin-style synthesizer with smooth frequency glides better represents:
- Upsweeps (rising frequency)
- Downsweeps (falling frequency)
- Complex modulated calls

The frequency range (50-500 Hz) is also lower than most piano notes, making a synthesizer more appropriate.

## Quick Start

### Option 1: Using the Run Script (Easiest)

```bash
cd OutreachProject
chmod +x run_whale_player.sh
./run_whale_player.sh
```

### Option 2: Manual Setup

```bash
cd OutreachProject

# Activate virtual environment
source ../Pytorch_scripts/.venv_py31018/bin/activate

# Install dependencies
pip install Flask scipy numpy matplotlib

# Run the application
python3 whale_player_app.py
```

Then open your browser to: **http://localhost:5000**

## How to Use

1. **Select a Call Type** - Click on one of the 7 call type buttons (color-coded by type)
2. **Load a Sample** - A random call will auto-load, or click "Random Sample" for another
3. **Play the Call** - Click "▶ Play Call" to hear the Theremin-style synthesis
4. **Watch the Visualization** - See the frequency contour and spectrogram
5. **Explore** - Try different call types to hear the diversity of bowhead vocalizations

## Call Types

| Type | Name | Description | Color |
|------|------|-------------|-------|
| 1 | Simple Upsweep | Rising frequency sweep | Blue |
| 2 | Simple Downsweep | Falling frequency sweep | Red |
| 3 | Constant Tone | Relatively constant frequency | Green |
| 4 | Complex Call | Modulated multi-component | Orange |
| 5 | Pulse Train | Repetitive pulse | Purple |
| 6 | Harmonic Call | Harmonic structure | Teal |
| 7 | Variable Call | Variable frequency pattern | Dark Orange |

## Technical Details

### Audio Synthesis
- **Oscillator Type**: Sine wave (most whale-like)
- **Frequency Mapping**: Extracts dominant frequency contour from spectrogram
- **Frequency Range**: 50-500 Hz (typical bowhead range)
- **Envelope**: Smooth attack/release for natural sound

### Data Processing
- Loads `.mat` files containing `SNR_gram` spectrograms
- Extracts weighted frequency centroid for each time step
- Generates visual spectrogram using matplotlib
- Serves data via Flask REST API

### Browser Compatibility
- Requires modern browser with Web Audio API support
- Tested on Chrome, Firefox, Safari, Edge

## File Structure

```
OutreachProject/
├── whale_player_app.py      # Flask backend
├── templates/
│   └── whale_player.html    # Web interface
├── run_whale_player.sh       # Quick start script
├── requirements.txt          # Python dependencies
└── README_WHALE_PLAYER.md    # This file
```

## Dependencies

- **Flask**: Web framework
- **scipy**: MATLAB file loading
- **numpy**: Numerical processing
- **matplotlib**: Spectrogram visualization

## Dataset

Data source: `/Users/oboulais/Public/Bowhead_DL_Project/BCB_Whale_Datasets/Unsupervised_database_Manual_100K_Y08101214.dir`

- ~70,000 manually classified calls
- Years: 2008, 2010, 2012, 2014
- Location: Alaskan Beaufort Sea
- Recording system: Directional autonomous recording packages (DARPs)

## Acknowledgments

Built on data and methods from:
- Thode et al. (2012) - Automated detection and localization of bowhead whale sounds
- Project 4509 (2410) - Development of deep learning bowhead whale call detector

## Future Enhancements

- [ ] Keyboard controls for triggering calls
- [ ] MIDI export of frequency contours
- [ ] Real-time pitch modification
- [ ] Call comparison mode
- [ ] Clustering visualization integration
- [ ] Educational mode with call descriptions

## License

For research and educational use. Please cite appropriately when using this tool or the underlying dataset.

---

*Created for outreach and education about bowhead whale bioacoustics*
