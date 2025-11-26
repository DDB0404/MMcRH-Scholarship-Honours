# Earnings Call Audio → Volatility Prediction Pipeline

**Created:** 2025-11-23
**Status:** ✅ Fully Functional & Tested

---

## Overview

This systematic pipeline processes earnings call audio and transcripts into features for predicting stock volatility. It combines emotion recognition, speech analysis, and financial market data into a complete machine learning workflow.

**Pipeline Flow:**
```
.mp3 + .json → audio processing → transcription → speaker alignment →
emotion extraction → feature engineering → volatility targets → ML training
```

---

## Files Created

### 1. Core Pipeline Module
**Location:** `src/earnings_pipeline.py` (1,000+ lines)

**Components:**
- `AudioProcessor` - MP3→WAV conversion, audio slicing
- `TranscriptParser` - Speaker turn extraction from Seeking Alpha JSON
- `WhisperAligner` - WhisperX transcription + fuzzy speaker matching
- `EmotionAnalyzer` - wav2vec2 emotion recognition (valence/arousal/dominance)
- `FeatureEngineer` - Speaker/call-level feature aggregation
- `VolatilityComputer` - yfinance integration + realized volatility computation

**Main Functions:**
```python
process_single_call(ticker, quarter, year, mp3_path, json_path, ...)
process_multiple_calls(calls_list, output_dir, ...)
```

### 2. Demonstration Notebook
**Location:** `notebooks/earnings_volatility_pipeline_demo.ipynb`

**Sections:**
1. Setup & Configuration
2. Single Call Demo (NVDA Q2 2026)
3. Batch Processing (Multiple Calls)
4. XGBoost Training Example
5. Time Series Cross-Validation

### 3. Test Script
**Location:** `test_pipeline.py`

Automated testing script that verifies pipeline correctness on NVDA Q2 2026.

---

## Test Results ✅

**Test Call:** NVDA Q2 2026 (Event: 2025-08-27)

**Processing Stats:**
- 143 audio segments processed
- 10 unique speakers identified
- 29 features extracted per call

**Realized Volatility Targets (Annualized):**
- 1-day: 0.1255 (12.55%)
- 3-day: 0.3660 (36.60%)
- 5-day: 0.2868 (28.68%)
- 7-day: 0.2965 (29.65%)
- 30-day: 0.3010 (30.10%)

✅ All values are realistic for NVDA around earnings dates.

**Output Files:**
- `data/processed_features/NVDA/Q2_2026/NVDA_Q2_2026_features.csv`
- `data/processed_features/NVDA/Q2_2026/final_NVDA_Q2_2026_segments.json`
- `data/processed_features/NVDA/Q2_2026/slices/` (143 WAV files)

---

## Quick Start

### 1. Process a Single Call

```python
from earnings_pipeline import process_single_call, get_default_config

config = get_default_config(
    hf_cache="/Volumes/Elements/huggingface_cache",
    whisper_cache="/Volumes/Elements/whisper_cache"
)

features_df = process_single_call(
    ticker="NVDA",
    quarter="Q2",
    year="2026",
    mp3_path="data/earnings-calls/NVDA/Q2_2026/NVDA_Q2_2026.mp3",
    transcript_json_path="data/earnings-transcripts/NVDA/Q2_2026/NVDA_Q2_2026.json",
    output_dir="data/processed_features",
    event_date="2025-08-27",
    compute_volatility=True,
    config=config
)

print(features_df.head())
```

### 2. Process Multiple Calls

```python
from earnings_pipeline import process_multiple_calls

calls = [
    {
        "ticker": "NVDA",
        "quarter": "Q2",
        "year": "2026",
        "mp3_path": "data/earnings-calls/NVDA/Q2_2026/NVDA_Q2_2026.mp3",
        "transcript_json_path": "data/earnings-transcripts/NVDA/Q2_2026/NVDA_Q2_2026.json",
        "event_date": "2025-08-27"
    },
    # Add more calls...
]

combined_df = process_multiple_calls(
    calls_list=calls,
    output_dir="data/processed_features",
    compute_volatility=True,
    config=config
)

# Save combined dataset
combined_df.to_csv("data/processed_features/all_calls.csv", index=False)
```

### 3. Run the Demo Notebook

```bash
jupyter notebook notebooks/earnings_volatility_pipeline_demo.ipynb
```

The notebook provides a complete walkthrough with visualizations.

---

## Feature Schema

### Speaker-Level Features (13 features):

| Feature | Description | Type |
|---------|-------------|------|
| `valence_mean` | Avg emotional positivity | float |
| `valence_std` | Emotional variability | float |
| `arousal_mean` | Avg energy/activation | float |
| `arousal_std` | Energy variability | float |
| `dominance_mean` | Avg confidence/control | float |
| `dominance_std` | Confidence variability | float |
| `duration_mean` | Avg segment duration (sec) | float |
| `duration_std` | Duration variability | float |
| `volatility` | Emotional energy volatility | float |
| `speed_mean` | Avg words per second | float |
| `speed_std` | Speech rate variability | float |
| `num_words_mean` | Avg words per segment | float |
| `num_words_std` | Word count variability | float |

### Call-Level Features (11 features):

| Feature | Description | Type |
|---------|-------------|------|
| `valence_mean` | Overall emotional positivity | float |
| `arousal_mean` | Overall energy level | float |
| `dominance_mean` | Overall confidence | float |
| `speed_mean` | Overall speech rate | float |
| `emotional_volatility` | Overall emotion variability | float |
| `total_duration` | Total call duration (sec) | float |
| `num_speakers` | Number of unique speakers | int |

### Volatility Targets (5 targets):

| Target | Description | Range |
|--------|-------------|-------|
| `rv_1d` | 1-day forward realized vol (annualized) | 0.0-1.0+ |
| `rv_3d` | 3-day forward realized vol (annualized) | 0.0-1.0+ |
| `rv_5d` | 5-day forward realized vol (annualized) | 0.0-1.0+ |
| `rv_7d` | 7-day forward realized vol (annualized) | 0.0-1.0+ |
| `rv_30d` | 30-day forward realized vol (annualized) | 0.0-1.0+ |

### Metadata (5 fields):

- `speaker` - Speaker name or "CALL_LEVEL"
- `ticker` - Stock ticker (e.g., "NVDA")
- `quarter` - Quarter (e.g., "Q2")
- `year` - Year (e.g., "2026")
- `event_date` - Event date (YYYY-MM-DD)

---

## Dependencies

**Required Packages:**
```
librosa==0.11.0
pydub==0.25.1
torch==2.8.0
transformers==4.57.1
rapidfuzz==3.14.1
yfinance==0.2.55
xgboost==3.0.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

**Optional:**
```
whisperx  # For transcription (can use cached transcriptions)
```

**Models Used:**
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` - Emotion recognition
- `whisperx/medium` - Speech-to-text transcription

---

## Configuration

### Default Configuration

```python
config = PipelineConfig(
    # Audio
    target_sample_rate=16000,
    audio_channels=1,

    # WhisperX
    whisper_model="medium",
    whisper_device="cpu",
    whisper_compute_type="int8",

    # Alignment
    max_gap_seconds=2.0,
    max_block_seconds=45.0,
    min_match_score=35,

    # Emotion
    emotion_model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",

    # Volatility
    vol_horizons=[1, 3, 5, 7, 30],
    annualize_vol=True,
    trading_days_per_year=252,

    # Caches
    hf_cache="/Volumes/Elements/huggingface_cache",
    whisper_cache="/Volumes/Elements/whisper_cache"
)
```

---

## Pipeline Architecture

### Phase 1: Audio Preprocessing
- Convert MP3 → WAV (16kHz mono)
- Standardize audio format

### Phase 2: Transcript Processing
- Parse Seeking Alpha JSON
- Extract speaker turns using regex
- Clean and normalize text

### Phase 3: Alignment
- Transcribe audio with WhisperX (or load cache)
- Merge short segments into blocks
- Fuzzy-match blocks to speakers
- Assign speakers to original segments

### Phase 4: Audio Slicing
- Slice audio file by timestamps
- Export individual segment WAV files

### Phase 5: Emotion Recognition
- Load wav2vec2 emotion model
- Extract valence/arousal/dominance for each segment
- Handle errors gracefully (NaN for failed extractions)

### Phase 6: Feature Engineering
- Compute segment-level derived features
- Aggregate by speaker (mean/std)
- Aggregate at call level (overall statistics)

### Phase 7: Volatility Computation
- Download price data via yfinance
- Compute log returns
- Calculate forward realized volatility for each horizon
- Annualize using sqrt(252/h) scaling

### Phase 8: Output
- Save features as CSV
- Save intermediate artifacts (JSON, audio slices)

---

## Performance

**Processing Time (NVDA Q2 2026):**
- Audio slicing: ~1 second
- Emotion extraction: ~5-8 minutes (143 segments × 2-3 sec/segment)
- Volatility computation: ~2 seconds
- **Total:** ~6-9 minutes per call (with cached transcription)

**Scalability:**
- Batch processing supported
- Can process multiple calls in sequence
- Caching prevents re-computation

---

## Next Steps

### 1. Scale Up Data Collection
- Process all available NVDA calls (Q1-Q4 across multiple years)
- Add other mega-cap stocks (AAPL, MSFT, GOOGL, AMZN, META)
- Target 20-50+ calls for robust ML models

### 2. Enhanced Feature Engineering
- Add textual sentiment from transcripts (BERT/FinBERT)
- Include financial metrics (revenue, EPS, guidance beats/misses)
- Compute relative changes vs previous quarter
- Speaker interaction features (interruptions, Q&A dynamics)

### 3. Model Improvements
- Hyperparameter tuning (Optuna/GridSearchCV)
- Ensemble methods (stacking, blending)
- Multi-task learning (predict multiple horizons jointly)
- Incorporate macroeconomic features (VIX, sector ETF vol)

### 4. Production Deployment
- Real-time processing pipeline for new earnings calls
- REST API for volatility predictions
- Integration with trading strategies
- Backtesting framework

---

## Troubleshooting

### Common Issues

**1. "No module named 'librosa'"**
```bash
pip install librosa pydub transformers torch rapidfuzz yfinance xgboost
```

**2. "MP3 file not found"**
- Check file paths match directory structure: `earnings-calls/{ticker}/{quarter}_{year}/{ticker}_{quarter}_{year}.mp3`

**3. "Emotion extraction error: no attribute 'logits'"**
- Ensure using `AutoModelForAudioClassification` not `Wav2Vec2Model`
- Fixed in `src/earnings_pipeline.py`

**4. "No trading days after event date"**
- Event date might be in the future or on a weekend/holiday
- yfinance may not have recent data yet

---

## Citation

If you use this pipeline in your research, please cite:

```
MMcRH Scholarship Honours Project
Earnings Call Audio Emotion Analysis for Stock Volatility Prediction
November 2025
```

---

## License

This pipeline is part of the MMcRH Scholarship Honours research project.

---

**Questions? Issues?**
Check the demo notebook or test script for examples of proper usage.
