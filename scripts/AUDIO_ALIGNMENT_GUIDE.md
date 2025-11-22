# Audio-Transcript Alignment Guide

## Overview

The `align_audio_transcript.py` script aligns earnings call audio (MP3) with transcripts (JSON), identifies speakers, and splits audio into sections and speaker segments.

## Features

- **Whisper Transcription**: Word-level timestamps for all speech
- **Speaker Diarization**: Identifies unique voices (pyannote.audio)
- **Speaker Matching**: Maps voice signatures to speaker names
- **Section Detection**: Splits into "Prepared Remarks" and "Q&A"
- **Multiple Outputs**: Aligned JSON, split MP3s, individual speaker segments

## Setup

### 1. Install System Dependencies

**macOS**:
```bash
brew install ffmpeg
```

**Linux**:
```bash
sudo apt-get install ffmpeg
```

### 2. Install Python Dependencies

```bash
cd /Users/dominicbyrne/MMcRH-Scholarship-Honours
pip install -r requirements.txt
```

This installs:
- faster-whisper (speech-to-text)
- pyannote.audio (speaker diarization)
- pydub (audio manipulation)
- torch, torchaudio (ML backends)
- fuzzywuzzy (text matching)

### 3. Setup HuggingFace Access

**Required for Speaker Diarization**

1. Create HuggingFace account: https://huggingface.co/join
2. Get your access token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "read" permissions
3. Accept model terms:
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"
4. Set your token:

```bash
export HF_TOKEN="your_token_here"
```

Or create `~/.bashrc` entry:
```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### Audio File Storage

**Audio files are stored on HuggingFace** (`baws004/nvda-audio`) to save local disk space. Use the `--use-hf` flag to automatically download them during processing.

### Process from HuggingFace (Recommended)

**Process a Single Call:**
```bash
python scripts/align_audio_transcript.py --file Q4_2025 --use-hf
```

**Process All Available Calls:**
```bash
python scripts/align_audio_transcript.py --all --use-hf
```

Files will be downloaded to `data/.cache/` and reused on subsequent runs.

### Process from Local Files

If you have audio files stored locally in `data/earnings-calls/NVDA/`:

```bash
python scripts/align_audio_transcript.py --all
```

### HuggingFace Token

The HF token can be provided via environment variable or command line:

```bash
# Set once
export HF_TOKEN="your_token_here"

# Then run
python scripts/align_audio_transcript.py --all --use-hf

# Or pass directly
python scripts/align_audio_transcript.py --all --use-hf --hf-token YOUR_TOKEN
```

## Output Structure

### 1. Aligned JSON (`data/aligned-transcripts/NVDA/`)

**File**: `NVDA_Q4_2025_aligned.json`

```json
{
  "fiscal_period": "Q4_2025",
  "audio_duration": 3628.5,
  "speakers": {
    "Jensen Huang": "President and CEO",
    "Colette Kress": "Executive VP & CFO",
    "Operator": "Conference Operator"
  },
  "segments": [
    {
      "start": 0.0,
      "end": 15.3,
      "duration": 15.3,
      "speaker": "Operator",
      "speaker_role": "Conference Operator",
      "speaker_id": "SPEAKER_00",
      "text": "Good afternoon. Welcome to NVIDIA's...",
      "section": "prepared_remarks"
    },
    {
      "start": 15.3,
      "end": 125.8,
      "duration": 110.5,
      "speaker": "Jensen Huang",
      "speaker_role": "President and CEO",
      "speaker_id": "SPEAKER_01",
      "text": "Thank you. This quarter was remarkable...",
      "section": "prepared_remarks"
    }
  ],
  "sections": {
    "prepared_remarks": {"start": 0, "end": 1850},
    "qa": {"start": 1850, "end": 3628.5}
  }
}
```

### 2. Split MP3 Files (`data/earnings-calls-split/NVDA/`)

- `NVDA_Q4_2025_prepared_remarks.mp3` - Company presentation section
- `NVDA_Q4_2025_qa.mp3` - Q&A section

### 3. Speaker Segments (`data/speaker-segments/NVDA/Q4_2025/`)

Individual MP3 files with all segments for each speaker concatenated:
- `jensen_huang.mp3` - All of CEO's speaking time
- `colette_kress.mp3` - All of CFO's speaking time
- `operator.mp3` - All operator segments
- `analyst_segments.mp3` - Various analyst questions

## Processing Time

Approximate times per 60-minute audio file:
- Whisper transcription: 2-5 minutes
- Speaker diarization: 5-10 minutes
- Matching & output: 1-2 minutes
- **Total**: ~10-15 minutes per file

**Performance Tips**:
- GPU (CUDA) is 5-10x faster than CPU
- Smaller Whisper model (`tiny`, `base`) is faster but less accurate
- Can process multiple files sequentially with `--all`

## Model Options

Edit `align_audio_transcript.py` to change models:

```python
# Line 29
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large-v2
```

**Model Trade-offs**:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy, slower
- `medium`/`large-v2`: Best accuracy, much slower

## Troubleshooting

### "No module named 'pyannote'"
```bash
pip install pyannote.audio
```

### "HuggingFace token required"
Set `HF_TOKEN` environment variable or use `--hf-token` parameter

### "OSError: cannot load library 'libsndfile'"
```bash
# macOS
brew install libsndfile

# Linux
sudo apt-get install libsndfile1
```

### "CUDA out of memory"
- Use CPU instead of GPU (automatic fallback)
- Or use smaller Whisper model
- Or process one file at a time

### "Speaker matching accuracy is low"
- Check that transcript has correct speaker names
- Transcript structure should have "Company Participants" section
- Audio quality may affect diarization

### "Q&A section not detected"
- Script uses heuristics to find Q&A start
- May need manual adjustment for non-standard call formats
- Check aligned JSON `sections` field

## Advanced Usage

### Process Specific Quarters

```bash
# Process multiple specific periods
for period in Q4_2025 Q1_2026 Q2_2026; do
    python scripts/align_audio_transcript.py --file $period
done
```

### Custom Output Processing

The aligned JSON can be used for further analysis:

```python
import json

# Load aligned data
with open('data/aligned-transcripts/NVDA/NVDA_Q4_2025_aligned.json') as f:
    data = json.load(f)

# Find all CEO statements
ceo_segments = [s for s in data['segments'] if 'CEO' in s['speaker_role']]

# Calculate speaking time per person
from collections import defaultdict
speaking_time = defaultdict(float)
for seg in data['segments']:
    speaking_time[seg['speaker']] += seg['duration']

print(f"Jensen Huang spoke for {speaking_time['Jensen Huang']/60:.1f} minutes")
```

## Directory Structure After Processing

```
MMcRH-Scholarship-Honours/
├── data/
│   ├── earnings-calls/NVDA/          # Original MP3s
│   │   ├── NVDA_Q4_2025.mp3
│   │   ├── NVDA_Q1_2026.mp3
│   │   └── NVDA_Q2_2026.mp3
│   ├── earnings-transcripts/NVDA/    # Original transcripts
│   │   ├── NVDA_Q4_2025.json
│   │   ├── NVDA_Q1_2026.json
│   │   └── NVDA_Q2_2026.json
│   ├── aligned-transcripts/NVDA/     # NEW: Aligned data
│   │   ├── NVDA_Q4_2025_aligned.json
│   │   ├── NVDA_Q1_2026_aligned.json
│   │   └── NVDA_Q2_2026_aligned.json
│   ├── earnings-calls-split/NVDA/    # NEW: Split by section
│   │   ├── NVDA_Q4_2025_prepared_remarks.mp3
│   │   ├── NVDA_Q4_2025_qa.mp3
│   │   ├── NVDA_Q1_2026_prepared_remarks.mp3
│   │   └── ...
│   └── speaker-segments/NVDA/        # NEW: By speaker
│       ├── Q4_2025/
│       │   ├── jensen_huang.mp3
│       │   ├── colette_kress.mp3
│       │   └── ...
│       ├── Q1_2026/
│       └── Q2_2026/
└── scripts/
    ├── align_audio_transcript.py
    └── AUDIO_ALIGNMENT_GUIDE.md
```

## Verification

Check that processing worked:

```bash
# List aligned JSONs
ls -lh data/aligned-transcripts/NVDA/

# List split MP3s
ls -lh data/earnings-calls-split/NVDA/

# List speaker segments
ls -lh data/speaker-segments/NVDA/Q4_2025/

# Check aligned JSON content
python -c "import json; d=json.load(open('data/aligned-transcripts/NVDA/NVDA_Q4_2025_aligned.json')); print(f'Segments: {len(d[\"segments\"])}, Speakers: {len(d[\"speakers\"])}')"
```

## Next Steps

After alignment, you can:
1. Analyze speaking patterns and time distribution
2. Extract specific speaker quotes with exact timestamps
3. Train voice recognition models on labeled speaker data
4. Create searchable database of who said what when
5. Compare CEO messaging across quarters
6. Analyze analyst question topics and frequency

## Support

For issues or questions about the alignment process:
1. Check this guide's Troubleshooting section
2. Verify all dependencies are installed correctly
3. Ensure audio and transcript files exist
4. Check that HuggingFace token has proper permissions
