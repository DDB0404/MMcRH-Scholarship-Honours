"""
Systematic Earnings Call Audio → Volatility Prediction Pipeline

This module provides a complete pipeline for processing earnings call audio and transcripts
into features for volatility prediction models.

Pipeline: .mp3 + .json → emotion features + realized volatility targets

Author: Scholarship Pipeline
Date: 2025-11-23
"""

import os
import re
import json
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import librosa
import torch
from pydub import AudioSegment
from transformers import AutoProcessor, AutoModelForAudioClassification
from rapidfuzz import fuzz
import yfinance as yf
from tqdm import tqdm

# Optional WhisperX import - gracefully handle if not available
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    print("Warning: WhisperX not available. Transcription features will be disabled.")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the earnings call processing pipeline"""

    # Audio settings
    target_sample_rate: int = 16000
    audio_channels: int = 1

    # WhisperX settings
    whisper_model: str = "medium"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Segment merging settings
    max_gap_seconds: float = 2.0
    max_block_seconds: float = 45.0
    min_match_score: int = 35

    # Emotion model
    emotion_model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    # Volatility settings
    vol_horizons: List[int] = None
    annualize_vol: bool = True
    trading_days_per_year: int = 252

    # Cache directories
    hf_cache: Optional[str] = None
    whisper_cache: Optional[str] = None

    def __post_init__(self):
        if self.vol_horizons is None:
            self.vol_horizons = [1, 3, 5, 7, 30]

        # Set environment variables for caches
        if self.hf_cache:
            os.environ["HF_HOME"] = self.hf_cache
        if self.whisper_cache:
            os.environ["CTRANSLATE2_HOME"] = os.path.join(self.whisper_cache, "ctranslate2")
            os.environ["TMPDIR"] = os.path.join(self.whisper_cache, "tmp")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_text(text: str) -> str:
    """Clean text for fuzzy matching: lowercase, remove punctuation and extra whitespace"""
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def extract_event_date(transcript_json: Dict) -> Optional[str]:
    """
    Extract event date from transcript JSON.

    Looks for publish_date or parses from title.
    Returns date string in YYYY-MM-DD format.
    """
    # Try publish_date field
    if "publish_date" in transcript_json and transcript_json["publish_date"]:
        try:
            # Parse various date formats
            date_str = transcript_json["publish_date"]
            # Try common formats
            for fmt in ["%Y-%m-%d", "%B %d, %Y", "%m/%d/%Y", "%d/%m/%Y"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime("%Y-%m-%d")
                except ValueError:
                    continue
        except Exception:
            pass

    # Try to parse from title if available
    if "title" in transcript_json:
        title = transcript_json["title"]
        # Look for patterns like "August 27, 2025" or similar
        date_patterns = [
            r"(\w+ \d{1,2}, \d{4})",  # August 27, 2025
            r"(\d{4}-\d{2}-\d{2})",   # 2025-08-27
        ]
        for pattern in date_patterns:
            match = re.search(pattern, title)
            if match:
                date_str = match.group(1)
                for fmt in ["%B %d, %Y", "%Y-%m-%d"]:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        return dt.strftime("%Y-%m-%d")
                    except ValueError:
                        continue

    return None


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    """Handles audio file conversion and slicing"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> str:
        """
        Convert MP3 to WAV with standardized format (16kHz, mono).

        Args:
            mp3_path: Path to input MP3 file
            wav_path: Path to output WAV file

        Returns:
            Path to created WAV file
        """
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(self.config.target_sample_rate)
        audio = audio.set_channels(self.config.audio_channels)
        audio.export(wav_path, format="wav")
        return wav_path

    def slice_audio_segments(
        self,
        audio_path: str,
        segments: List[Dict],
        output_dir: str,
        segment_key: str = "audio_file"
    ) -> List[Dict]:
        """
        Slice audio file into segments and save individual WAV files.

        Args:
            audio_path: Path to full audio file
            segments: List of segment dicts with 'start' and 'end' timestamps
            output_dir: Directory to save audio slices
            segment_key: Key to add to each segment dict with the slice path

        Returns:
            Updated segments list with audio file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        audio = AudioSegment.from_wav(audio_path)

        for i, seg in enumerate(tqdm(segments, desc="Slicing audio")):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)

            slice_filename = f"seg_{i:04d}.wav"
            slice_path = os.path.join(output_dir, slice_filename)

            audio_slice = audio[start_ms:end_ms]
            audio_slice.export(slice_path, format="wav")

            seg[segment_key] = slice_path

        return segments


# ============================================================================
# TRANSCRIPT PARSER
# ============================================================================

class TranscriptParser:
    """Parse Seeking Alpha transcript JSON to extract speaker turns"""

    # Regex pattern for valid speaker names
    VALID_SPEAKER_RE = re.compile(
        r"^(Operator|"
        r"[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|"
        r"[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+|"
        r"[A-Z]\.\s+[A-Z][a-z]+\s+[A-Z][a-z]+)$"
    )

    def extract_speaker_turns(self, transcript_json_path: str) -> List[Dict[str, str]]:
        """
        Extract speaker turns from Seeking Alpha transcript JSON.

        Args:
            transcript_json_path: Path to transcript JSON file

        Returns:
            List of dicts with keys: speaker, text, clean_text
        """
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        full_text = data.get("transcript", "")
        if not full_text:
            raise ValueError("No transcript field found in JSON")

        lines = full_text.split("\n")
        turns = []
        current_speaker = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a speaker name
            if self.VALID_SPEAKER_RE.match(line):
                # Save previous speaker's turn
                if current_speaker and current_text:
                    text = " ".join(current_text)
                    turns.append({
                        "speaker": current_speaker,
                        "text": text,
                        "clean_text": clean_text(text)
                    })

                # Start new speaker turn
                current_speaker = line
                current_text = []
            else:
                # Add to current speaker's text
                if current_speaker:
                    current_text.append(line)

        # Don't forget the last speaker
        if current_speaker and current_text:
            text = " ".join(current_text)
            turns.append({
                "speaker": current_speaker,
                "text": text,
                "clean_text": clean_text(text)
            })

        return turns


# ============================================================================
# WHISPER ALIGNER
# ============================================================================

class WhisperAligner:
    """Align WhisperX transcription with speaker turns using fuzzy matching"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def load_model(self):
        """Lazy load WhisperX model"""
        if not WHISPERX_AVAILABLE:
            raise RuntimeError("WhisperX is not available")

        if self.model is None:
            self.model = whisperx.load_model(
                self.config.whisper_model,
                self.config.whisper_device,
                compute_type=self.config.whisper_compute_type
            )
        return self.model

    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio using WhisperX.

        Args:
            audio_path: Path to WAV file

        Returns:
            Transcription dict with segments
        """
        model = self.load_model()
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio)
        return result

    def merge_segments(
        self,
        segments: List[Dict],
        max_gap: Optional[float] = None,
        max_block: Optional[float] = None
    ) -> List[Dict]:
        """
        Merge short WhisperX segments into longer blocks.

        Args:
            segments: List of WhisperX segments
            max_gap: Maximum gap between segments to merge (seconds)
            max_block: Maximum block duration (seconds)

        Returns:
            List of merged blocks
        """
        if max_gap is None:
            max_gap = self.config.max_gap_seconds
        if max_block is None:
            max_block = self.config.max_block_seconds

        if not segments:
            return []

        blocks = []
        current_block = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"]
        }

        for seg in segments[1:]:
            gap = seg["start"] - current_block["end"]
            duration = seg["end"] - current_block["start"]

            if gap <= max_gap and duration <= max_block:
                # Merge into current block
                current_block["end"] = seg["end"]
                current_block["text"] += " " + seg["text"]
            else:
                # Save current block and start new one
                blocks.append(current_block)
                current_block = {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }

        # Add final block
        blocks.append(current_block)
        return blocks

    def match_block_to_speaker(
        self,
        block_text: str,
        speaker_turns: List[Dict],
        min_score: Optional[int] = None
    ) -> Tuple[Optional[str], float]:
        """
        Match a text block to a speaker turn using fuzzy matching.

        Args:
            block_text: Text to match
            speaker_turns: List of speaker turn dicts with 'speaker' and 'clean_text'
            min_score: Minimum match score (0-100)

        Returns:
            Tuple of (speaker_name, confidence_score) or (None, 0) if no match
        """
        if min_score is None:
            min_score = self.config.min_match_score

        cleaned_block = clean_text(block_text)

        best_speaker = None
        best_score = 0

        for turn in speaker_turns:
            score = fuzz.partial_ratio(cleaned_block, turn["clean_text"])
            if score > best_score:
                best_score = score
                best_speaker = turn["speaker"]

        if best_score >= min_score:
            return best_speaker, best_score
        else:
            return None, 0

    def align_segments_with_speakers(
        self,
        whisper_segments: List[Dict],
        speaker_turns: List[Dict]
    ) -> List[Dict]:
        """
        Full alignment pipeline: merge segments → match to speakers → assign to original segments.

        Args:
            whisper_segments: WhisperX transcription segments
            speaker_turns: Speaker turns from transcript parser

        Returns:
            Original segments with added 'speaker' and 'speaker_confidence' fields
        """
        # Step 1: Merge segments into blocks
        blocks = self.merge_segments(whisper_segments)

        # Step 2: Match blocks to speakers
        for block in blocks:
            speaker, confidence = self.match_block_to_speaker(
                block["text"],
                speaker_turns
            )
            block["speaker"] = speaker
            block["speaker_confidence"] = confidence

        # Step 3: Assign speakers back to original segments
        aligned_segments = []
        for seg in whisper_segments:
            seg_mid = (seg["start"] + seg["end"]) / 2

            # Find which block this segment belongs to
            assigned_speaker = None
            assigned_confidence = 0

            for block in blocks:
                if block["start"] <= seg_mid <= block["end"]:
                    assigned_speaker = block["speaker"]
                    assigned_confidence = block["speaker_confidence"]
                    break

            aligned_segments.append({
                **seg,
                "speaker": assigned_speaker,
                "speaker_confidence": assigned_confidence
            })

        return aligned_segments


# ============================================================================
# EMOTION ANALYZER
# ============================================================================

class EmotionAnalyzer:
    """Extract emotion features from audio using wav2vec2 model"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.processor = None

    def load_model(self):
        """Lazy load emotion recognition model"""
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(
                self.config.emotion_model_name
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.config.emotion_model_name
            )
            self.model.eval()
        return self.model, self.processor

    def extract_emotion_from_audio(self, wav_path: str) -> Dict[str, float]:
        """
        Extract emotion dimensions from audio file.

        Args:
            wav_path: Path to WAV audio file

        Returns:
            Dict with keys: valence, arousal, dominance
        """
        model, processor = self.load_model()

        try:
            # Load audio
            audio, sr = librosa.load(wav_path, sr=self.config.target_sample_rate)

            # Process through model
            inputs = processor(
                audio,
                sampling_rate=self.config.target_sample_rate,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            # Extract emotion scores
            scores = outputs.logits[0].cpu().numpy().tolist()

            return {
                "valence": scores[0],
                "arousal": scores[1],
                "dominance": scores[2]
            }
        except Exception as e:
            print(f"Error extracting emotion from {wav_path}: {e}")
            return {
                "valence": np.nan,
                "arousal": np.nan,
                "dominance": np.nan
            }

    def extract_emotions_for_segments(
        self,
        segments: List[Dict],
        audio_file_key: str = "audio_file"
    ) -> List[Dict]:
        """
        Extract emotions for all segments in a list.

        Args:
            segments: List of segment dicts with audio file paths
            audio_file_key: Key in segment dict containing audio file path

        Returns:
            Segments with added emotion fields: valence, arousal, dominance
        """
        for seg in tqdm(segments, desc="Extracting emotions"):
            if audio_file_key in seg:
                emotions = self.extract_emotion_from_audio(seg[audio_file_key])
                seg.update(emotions)
            else:
                seg.update({
                    "valence": np.nan,
                    "arousal": np.nan,
                    "dominance": np.nan
                })

        return segments


# ============================================================================
# FEATURE ENGINEER
# ============================================================================

class FeatureEngineer:
    """Aggregate segment-level data into speaker and call-level features"""

    @staticmethod
    def compute_segment_features(segments: List[Dict]) -> pd.DataFrame:
        """
        Convert segments to DataFrame and compute derived features.

        Args:
            segments: List of segment dicts with emotions and text

        Returns:
            DataFrame with segment-level features
        """
        df = pd.DataFrame(segments)

        # Compute duration
        if "start" in df.columns and "end" in df.columns:
            df["duration"] = df["end"] - df["start"]

        # Compute emotional energy
        if all(col in df.columns for col in ["arousal", "dominance"]):
            df["emotional_energy"] = (df["arousal"].abs() + df["dominance"].abs()) / 2

        # Compute speech metrics
        if "text" in df.columns:
            df["num_words"] = df["text"].apply(lambda t: len(str(t).split()))
            if "duration" in df.columns:
                df["words_per_second"] = df["num_words"] / df["duration"].replace(0, np.nan)

        return df

    @staticmethod
    def aggregate_speaker_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate segment features by speaker.

        Args:
            df: DataFrame with segment-level features

        Returns:
            DataFrame with speaker-level aggregated features
        """
        if "speaker" not in df.columns:
            raise ValueError("DataFrame must have 'speaker' column")

        # Define aggregation functions
        agg_dict = {}

        # Emotion features
        for col in ["valence", "arousal", "dominance"]:
            if col in df.columns:
                agg_dict[col] = ["mean", "std"]

        # Duration
        if "duration" in df.columns:
            agg_dict["duration"] = ["mean", "std"]

        # Emotional energy volatility
        if "emotional_energy" in df.columns:
            agg_dict["emotional_energy"] = ["std"]

        # Speech metrics
        if "words_per_second" in df.columns:
            agg_dict["words_per_second"] = ["mean", "std"]

        if "num_words" in df.columns:
            agg_dict["num_words"] = ["mean", "std"]

        # Perform aggregation
        speaker_features = df.groupby("speaker").agg(agg_dict)

        # Flatten multi-level columns
        speaker_features.columns = [
            f"{col}_{stat}" if stat != "std" or col != "emotional_energy" else "volatility"
            for col, stat in speaker_features.columns
        ]

        # Rename for clarity
        if "emotional_energy_std" in speaker_features.columns:
            speaker_features.rename(columns={"emotional_energy_std": "volatility"}, inplace=True)

        # Add speaker count
        speaker_features["segment_count"] = df.groupby("speaker").size()

        return speaker_features.reset_index()

    @staticmethod
    def aggregate_call_level_features(df: pd.DataFrame) -> pd.Series:
        """
        Aggregate features at the call level (across all speakers).

        Args:
            df: DataFrame with segment-level features

        Returns:
            Series with call-level aggregated features
        """
        features = {}

        # Overall emotion statistics
        for col in ["valence", "arousal", "dominance"]:
            if col in df.columns:
                features[f"{col}_mean"] = df[col].mean()
                features[f"{col}_std"] = df[col].std()

        # Overall speech metrics
        if "words_per_second" in df.columns:
            features["speed_mean"] = df["words_per_second"].mean()
            features["speed_std"] = df["words_per_second"].std()

        if "emotional_energy" in df.columns:
            features["emotional_volatility"] = df["emotional_energy"].std()

        # Total duration
        if "duration" in df.columns:
            features["total_duration"] = df["duration"].sum()

        # Speaker diversity
        if "speaker" in df.columns:
            features["num_speakers"] = df["speaker"].nunique()

        return pd.Series(features)


# ============================================================================
# VOLATILITY COMPUTER
# ============================================================================

class VolatilityComputer:
    """Compute realized volatility from stock price data"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def download_price_data(
        self,
        ticker: str,
        event_date: str,
        buffer_days_before: int = 10,
        buffer_days_after: int = 60
    ) -> pd.DataFrame:
        """
        Download stock price data around event date.

        Args:
            ticker: Stock ticker symbol
            event_date: Event date in YYYY-MM-DD format
            buffer_days_before: Days of data to fetch before event
            buffer_days_after: Days of data to fetch after event

        Returns:
            DataFrame with price data and log returns
        """
        event_dt = pd.to_datetime(event_date)
        start = event_dt - timedelta(days=buffer_days_before)
        end = event_dt + timedelta(days=buffer_days_after)

        # Download data
        px = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False
        )

        if px.empty:
            raise ValueError(f"No price data available for {ticker}")

        # Extract close prices and compute log returns
        if "Close" in px.columns:
            px = px[["Close"]]
        elif ("Close", ticker) in px.columns:
            px = pd.DataFrame(px[("Close", ticker)], columns=["Close"])

        px["log_ret"] = np.log(px["Close"]).diff()
        px = px.dropna()

        return px

    def compute_realized_volatility(
        self,
        log_returns: pd.Series,
        start_date: pd.Timestamp,
        horizon: int
    ) -> float:
        """
        Compute realized volatility over a forward horizon.

        Formula: RV_h = sqrt(252/h * sum(r_t^2))

        Args:
            log_returns: Series of log returns indexed by date
            start_date: Start date for the forward window
            horizon: Number of trading days forward

        Returns:
            Annualized realized volatility
        """
        try:
            loc = log_returns.index.get_loc(start_date)
        except KeyError:
            return np.nan

        window = log_returns.iloc[loc:loc + horizon]

        if len(window) < horizon:
            return np.nan  # Insufficient data

        if self.config.annualize_vol:
            rv = np.sqrt(
                self.config.trading_days_per_year / horizon * np.sum(window ** 2)
            )
        else:
            rv = np.sqrt(np.sum(window ** 2) / horizon)

        return rv

    def compute_all_horizons(
        self,
        ticker: str,
        event_date: str,
        horizons: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute realized volatility for multiple horizons.

        Args:
            ticker: Stock ticker symbol
            event_date: Event date in YYYY-MM-DD format
            horizons: List of horizons (trading days). Uses config default if None.

        Returns:
            Dict with keys like 'rv_1d', 'rv_3d', etc.
        """
        if horizons is None:
            horizons = self.config.vol_horizons

        # Download price data
        try:
            px = self.download_price_data(ticker, event_date)
        except Exception as e:
            print(f"Error downloading price data for {ticker}: {e}")
            return {f"rv_{h}d": np.nan for h in horizons}

        # Find first trading day after event
        event_dt = pd.to_datetime(event_date)
        after_event = px.loc[px.index > event_dt]

        if after_event.empty:
            print(f"No trading days found after {event_date} for {ticker}")
            return {f"rv_{h}d": np.nan for h in horizons}

        t0 = after_event.index.min()
        log_returns = px["log_ret"]

        # Compute volatility for each horizon
        results = {}
        for h in horizons:
            rv = self.compute_realized_volatility(log_returns, t0, h)
            results[f"rv_{h}d"] = rv

        return results


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def process_single_call(
    ticker: str,
    quarter: str,
    year: str,
    mp3_path: str,
    transcript_json_path: str,
    output_dir: str,
    event_date: Optional[str] = None,
    compute_volatility: bool = True,
    config: Optional[PipelineConfig] = None,
    use_cached_transcription: bool = True,
    transcription_cache_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Complete pipeline: Process a single earnings call from audio + transcript to features + volatility.

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA')
        quarter: Quarter (e.g., 'Q2')
        year: Year (e.g., '2026')
        mp3_path: Path to MP3 audio file
        transcript_json_path: Path to Seeking Alpha transcript JSON
        output_dir: Directory to save outputs
        event_date: Event date in YYYY-MM-DD format (auto-extracted if None)
        compute_volatility: Whether to compute volatility targets
        config: Pipeline configuration (uses defaults if None)
        use_cached_transcription: Whether to use cached WhisperX transcription if available
        transcription_cache_path: Path to cached transcription JSON

    Returns:
        DataFrame with features and volatility targets
    """
    if config is None:
        config = PipelineConfig()

    print(f"\n{'='*80}")
    print(f"Processing {ticker} {quarter} {year}")
    print(f"{'='*80}\n")

    # Create output directory structure
    call_output_dir = os.path.join(output_dir, ticker, f"{quarter}_{year}")
    os.makedirs(call_output_dir, exist_ok=True)
    slices_dir = os.path.join(call_output_dir, "slices")

    # Initialize pipeline components
    audio_processor = AudioProcessor(config)
    transcript_parser = TranscriptParser()
    whisper_aligner = WhisperAligner(config)
    emotion_analyzer = EmotionAnalyzer(config)
    feature_engineer = FeatureEngineer()
    volatility_computer = VolatilityComputer(config)

    # Step 1: Convert MP3 to WAV
    print("[1/8] Converting MP3 to WAV...")
    wav_path = os.path.join(call_output_dir, f"{ticker}_{quarter}_{year}.wav")
    if not os.path.exists(wav_path):
        audio_processor.convert_mp3_to_wav(mp3_path, wav_path)
    else:
        print(f"  Using existing WAV: {wav_path}")

    # Step 2: Parse transcript for speaker turns
    print("[2/8] Parsing transcript for speaker turns...")
    speaker_turns = transcript_parser.extract_speaker_turns(transcript_json_path)
    print(f"  Found {len(speaker_turns)} speaker turns")

    # Extract event date if not provided
    if event_date is None:
        print("[3/8] Extracting event date from transcript...")
        with open(transcript_json_path, 'r') as f:
            transcript_json = json.load(f)
        event_date = extract_event_date(transcript_json)
        if event_date:
            print(f"  Event date: {event_date}")
        else:
            print("  Warning: Could not extract event date")

    # Step 3: Transcribe audio with WhisperX (or load cached)
    print("[4/8] Transcribing audio with WhisperX...")

    if transcription_cache_path is None:
        transcription_cache_path = os.path.join(
            call_output_dir,
            f"{ticker}_{quarter}_{year}_whisperx_transcription.json"
        )

    if use_cached_transcription and os.path.exists(transcription_cache_path):
        print(f"  Loading cached transcription: {transcription_cache_path}")
        with open(transcription_cache_path, 'r') as f:
            transcription = json.load(f)
    else:
        if not WHISPERX_AVAILABLE:
            raise RuntimeError(
                "WhisperX not available and no cached transcription found. "
                "Please install WhisperX or provide a cached transcription."
            )
        print(f"  Running WhisperX transcription...")
        transcription = whisper_aligner.transcribe_audio(wav_path)
        # Cache the transcription
        with open(transcription_cache_path, 'w') as f:
            json.dump(transcription, f, indent=2)

    whisper_segments = transcription.get("segments", [])
    print(f"  Found {len(whisper_segments)} WhisperX segments")

    # Step 4: Align segments with speakers
    print("[5/8] Aligning segments with speakers...")
    aligned_segments = whisper_aligner.align_segments_with_speakers(
        whisper_segments,
        speaker_turns
    )

    # Save aligned segments
    aligned_path = os.path.join(call_output_dir, f"final_{ticker}_{quarter}_{year}_segments.json")

    # Step 5: Slice audio
    print("[6/8] Slicing audio into segments...")
    aligned_segments = audio_processor.slice_audio_segments(
        wav_path,
        aligned_segments,
        slices_dir
    )

    # Save segments with audio paths
    with open(aligned_path, 'w') as f:
        json.dump(aligned_segments, f, indent=2)

    # Step 6: Extract emotions
    print("[7/8] Extracting emotion features...")
    segments_with_emotions = emotion_analyzer.extract_emotions_for_segments(
        aligned_segments
    )

    # Save segments with emotions
    emotions_path = os.path.join(
        call_output_dir,
        f"final_{ticker}_{quarter}_{year}_segments_with_emotions.json"
    )
    with open(emotions_path, 'w') as f:
        json.dump(segments_with_emotions, f, indent=2)

    # Step 7: Compute features
    print("[8/8] Computing aggregated features...")
    df_segments = feature_engineer.compute_segment_features(segments_with_emotions)

    # Aggregate by speaker
    speaker_features = feature_engineer.aggregate_speaker_features(df_segments)

    # Add metadata
    speaker_features["ticker"] = ticker
    speaker_features["quarter"] = quarter
    speaker_features["year"] = year
    speaker_features["event_date"] = event_date

    # Compute call-level features
    call_features = feature_engineer.aggregate_call_level_features(df_segments)

    # Add call-level features as a separate row with speaker="CALL_LEVEL"
    call_row = pd.DataFrame([{
        "speaker": "CALL_LEVEL",
        "ticker": ticker,
        "quarter": quarter,
        "year": year,
        "event_date": event_date,
        **call_features.to_dict()
    }])

    # Combine speaker and call-level features
    features_df = pd.concat([speaker_features, call_row], ignore_index=True)

    # Step 8: Compute volatility targets
    if compute_volatility and event_date:
        print("\n[Bonus] Computing realized volatility targets...")
        try:
            vol_targets = volatility_computer.compute_all_horizons(ticker, event_date)
            for key, value in vol_targets.items():
                features_df[key] = value
            print(f"  Volatility targets: {vol_targets}")
        except Exception as e:
            print(f"  Error computing volatility: {e}")
            for h in config.vol_horizons:
                features_df[f"rv_{h}d"] = np.nan

    # Save features
    features_path = os.path.join(
        call_output_dir,
        f"{ticker}_{quarter}_{year}_features.csv"
    )
    features_df.to_csv(features_path, index=False)
    print(f"\nFeatures saved to: {features_path}")

    return features_df


def process_multiple_calls(
    calls_list: List[Dict[str, str]],
    output_dir: str,
    compute_volatility: bool = True,
    config: Optional[PipelineConfig] = None
) -> pd.DataFrame:
    """
    Process multiple earnings calls and combine into a single dataset.

    Args:
        calls_list: List of dicts with keys:
            - ticker: Stock ticker
            - quarter: Quarter (e.g., 'Q2')
            - year: Year (e.g., '2026')
            - mp3_path: Path to MP3 file
            - transcript_json_path: Path to transcript JSON
            - event_date: Optional event date (YYYY-MM-DD)
            - transcription_cache_path: Optional path to cached transcription
        output_dir: Base output directory
        compute_volatility: Whether to compute volatility targets
        config: Pipeline configuration

    Returns:
        Combined DataFrame with all calls' features
    """
    if config is None:
        config = PipelineConfig()

    all_features = []

    for call_info in calls_list:
        try:
            features = process_single_call(
                ticker=call_info["ticker"],
                quarter=call_info["quarter"],
                year=call_info["year"],
                mp3_path=call_info["mp3_path"],
                transcript_json_path=call_info["transcript_json_path"],
                output_dir=output_dir,
                event_date=call_info.get("event_date"),
                compute_volatility=compute_volatility,
                config=config,
                transcription_cache_path=call_info.get("transcription_cache_path")
            )
            all_features.append(features)
        except Exception as e:
            print(f"\nERROR processing {call_info.get('ticker')} {call_info.get('quarter')} {call_info.get('year')}: {e}")
            continue

    if not all_features:
        raise ValueError("No calls were successfully processed")

    # Combine all features
    combined_df = pd.concat(all_features, ignore_index=True)

    # Save combined dataset
    combined_path = os.path.join(output_dir, "all_calls_features_with_volatility.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\n{'='*80}")
    print(f"Combined dataset saved to: {combined_path}")
    print(f"Total rows: {len(combined_df)}")
    print(f"{'='*80}\n")

    return combined_df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_default_config(
    hf_cache: str = "/Volumes/Elements/huggingface_cache",
    whisper_cache: str = "/Volumes/Elements/whisper_cache"
) -> PipelineConfig:
    """Get default pipeline configuration with custom cache directories"""
    return PipelineConfig(
        hf_cache=hf_cache,
        whisper_cache=whisper_cache
    )
