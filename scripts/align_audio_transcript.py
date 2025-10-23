#!/usr/bin/env python3
"""
Audio-Transcript Alignment and Speaker Segmentation
Aligns earnings call audio with transcripts, identifies speakers, and splits into sections
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Audio processing
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch

# Text processing
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Configuration
BASE_DIR = Path("/Users/dominicbyrne/MMcRH-Scholarship-Honours")
TRANSCRIPT_DIR = BASE_DIR / "data/earnings-transcripts/NVDA"
AUDIO_DIR = BASE_DIR / "data/earnings-calls/NVDA"
OUTPUT_ALIGNED_DIR = BASE_DIR / "data/aligned-transcripts/NVDA"
OUTPUT_SPLIT_DIR = BASE_DIR / "data/earnings-calls-split/NVDA"
OUTPUT_SPEAKER_DIR = BASE_DIR / "data/speaker-segments/NVDA"
CACHE_DIR = BASE_DIR / "data/.cache"  # For HuggingFace downloads

# HuggingFace dataset
HF_DATASET_REPO = "baws004/nvda-audio"

# Model settings
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large-v2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary output directories"""
    OUTPUT_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directories ready")


def download_from_hf(fiscal_period: str) -> Tuple[Path, Path]:
    """
    Download audio and transcript files from HuggingFace dataset
    Returns: (audio_path, transcript_path)
    """
    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading files from HuggingFace: {HF_DATASET_REPO}")

    # Download audio file
    audio_filename = f"NVDA_{fiscal_period}.mp3"
    audio_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"audio/{audio_filename}",
        repo_type="dataset",
        cache_dir=str(CACHE_DIR)
    )
    logger.info(f"Downloaded audio: {audio_filename}")

    # Download transcript file
    transcript_filename = f"NVDA_{fiscal_period}.json"
    transcript_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=f"transcripts/{transcript_filename}",
        repo_type="dataset",
        cache_dir=str(CACHE_DIR)
    )
    logger.info(f"Downloaded transcript: {transcript_filename}")

    return Path(audio_path), Path(transcript_path)


class TranscriptParser:
    """Parse earnings call transcript to extract speaker information and structure"""

    def __init__(self, transcript_json_path: Path):
        with open(transcript_json_path, 'r') as f:
            self.data = json.load(f)

        self.full_text = self.data['transcript']
        self.fiscal_period = self.data.get('fiscal_period', 'Unknown')

    def extract_speakers(self) -> Dict[str, str]:
        """
        Extract speaker names and their roles from transcript
        Returns: {speaker_name: role}
        """
        speakers = {}

        # Look for "Company Participants" section
        company_match = re.search(
            r'Company Participants(.*?)(?:Conference Call Participants|Operator)',
            self.full_text,
            re.DOTALL | re.IGNORECASE
        )

        if company_match:
            company_section = company_match.group(1)
            # Extract names and roles like "Jensen Huang - President and CEO"
            for line in company_section.split('\n'):
                match = re.match(r'([A-Z][^-]+)\s*-\s*(.+)', line.strip())
                if match:
                    name = match.group(1).strip()
                    role = match.group(2).strip()
                    speakers[name] = role

        # Look for "Conference Call Participants" (analysts)
        analyst_match = re.search(
            r'Conference Call Participants(.*?)(?:Operator|Presentation)',
            self.full_text,
            re.DOTALL | re.IGNORECASE
        )

        if analyst_match:
            analyst_section = analyst_match.group(1)
            for line in analyst_section.split('\n'):
                match = re.match(r'([A-Z][^-]+)\s*-\s*(.+)', line.strip())
                if match:
                    name = match.group(1).strip()
                    role = match.group(2).strip()
                    speakers[name] = f"Analyst - {role}"

        # Add Operator
        if 'Operator' in self.full_text:
            speakers['Operator'] = 'Conference Operator'

        logger.info(f"Found {len(speakers)} speakers in transcript")
        return speakers

    def detect_qa_start(self) -> Optional[int]:
        """
        Detect where Q&A section starts in the transcript
        Returns: Character position, or None if not found
        """
        # Common patterns for Q&A start
        patterns = [
            r'Question-and-Answer Session',
            r'Question[\s-]and[\s-]Answer',
            r'\[Operator Instructions\]',
            r'Operator.*We will now begin the question',
            r'Our first question comes from',
        ]

        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.IGNORECASE)
            if match:
                position = match.start()
                logger.info(f"Q&A section detected at character {position}")
                return position

        logger.warning("Could not detect Q&A section start")
        return None

    def extract_speaker_turns(self) -> List[Dict]:
        """
        Extract individual speaker turns from transcript
        Returns: List of {speaker, text, approx_position}
        """
        turns = []

        # Pattern to match speaker turns like "Jensen Huang\nThank you..."
        # or "Jensen Huang - President and CEO\nThank you..."
        pattern = r'([A-Z][^\n]+?)(?:\s*-\s*[^\n]+?)?\n([^\n]+(?:\n(?![A-Z][^\n]+?(?:\s*-\s*[^\n]+?)?\n)[^\n]+)*)'

        matches = re.finditer(pattern, self.full_text)

        for match in matches:
            speaker = match.group(1).strip()
            text = match.group(2).strip()

            # Clean up speaker name (remove role if present)
            speaker = re.sub(r'\s*-\s*.+$', '', speaker).strip()

            if len(text) > 20:  # Filter out very short matches
                turns.append({
                    'speaker': speaker,
                    'text': text,
                    'char_position': match.start()
                })

        logger.info(f"Extracted {len(turns)} speaker turns from transcript")
        return turns


class AudioProcessor:
    """Process audio: transcribe and perform speaker diarization"""

    def __init__(self, audio_path: Path, hf_token: Optional[str] = None):
        self.audio_path = audio_path
        self.hf_token = hf_token or os.getenv('HF_TOKEN')

        # Load audio
        logger.info(f"Loading audio from {audio_path}")
        self.audio = AudioSegment.from_mp3(str(audio_path))
        self.duration_seconds = len(self.audio) / 1000.0
        logger.info(f"Audio duration: {self.duration_seconds:.1f} seconds")

    def transcribe_with_whisper(self) -> List[Dict]:
        """
        Transcribe audio using Whisper with word-level timestamps
        Returns: List of {start, end, text}
        """
        logger.info(f"Transcribing audio with Whisper ({WHISPER_MODEL} model)...")

        model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

        segments, info = model.transcribe(
            str(self.audio_path),
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # Voice activity detection
        )

        transcription = []
        for segment in tqdm(segments, desc="Processing Whisper segments"):
            transcription.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': [
                    {'start': w.start, 'end': w.end, 'word': w.word}
                    for w in segment.words
                ] if hasattr(segment, 'words') and segment.words else []
            })

        logger.info(f"Transcribed {len(transcription)} segments")
        return transcription

    def diarize_speakers(self) -> List[Dict]:
        """
        Perform speaker diarization using pyannote.audio
        Returns: List of {start, end, speaker_id}
        """
        if not self.hf_token:
            logger.error("HuggingFace token required for speaker diarization!")
            logger.error("Set HF_TOKEN environment variable or pass hf_token parameter")
            return []

        logger.info("Performing speaker diarization with pyannote.audio...")

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )

            # Run on GPU if available
            if DEVICE == "cuda":
                pipeline.to(torch.device("cuda"))

            diarization = pipeline(str(self.audio_path))

            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker_id': speaker
                })

            logger.info(f"Identified {len(set(s['speaker_id'] for s in segments))} unique speakers")
            logger.info(f"Created {len(segments)} speaker segments")

            return segments

        except Exception as e:
            logger.error(f"Speaker diarization failed: {e}")
            logger.error("Make sure you've accepted the model terms at:")
            logger.error("https://huggingface.co/pyannote/speaker-diarization-3.1")
            return []


class SpeakerMatcher:
    """Match anonymous speaker IDs to real names using transcript"""

    def __init__(self, diarized_segments: List[Dict], whisper_segments: List[Dict],
                 transcript_speakers: Dict[str, str]):
        self.diarized_segments = diarized_segments
        self.whisper_segments = whisper_segments
        self.transcript_speakers = transcript_speakers

    def match_speakers(self) -> Dict[str, str]:
        """
        Match speaker_ids (SPEAKER_00, SPEAKER_01) to real names
        Returns: {speaker_id: real_name}
        """
        logger.info("Matching speaker IDs to real names...")

        # Get unique speaker IDs
        speaker_ids = list(set(seg['speaker_id'] for seg in self.diarized_segments))

        # For each speaker ID, collect their spoken text
        speaker_texts = {spk_id: [] for spk_id in speaker_ids}

        for diar_seg in self.diarized_segments:
            # Find overlapping Whisper segments
            overlapping_text = []
            for whisper_seg in self.whisper_segments:
                # Check if segments overlap
                if (whisper_seg['start'] < diar_seg['end'] and
                    whisper_seg['end'] > diar_seg['start']):
                    overlapping_text.append(whisper_seg['text'])

            if overlapping_text:
                speaker_texts[diar_seg['speaker_id']].extend(overlapping_text)

        # Match each speaker ID to a name by comparing texts
        speaker_mapping = {}
        used_names = set()

        for spk_id in sorted(speaker_ids):
            if not speaker_texts[spk_id]:
                continue

            # Combine texts for this speaker
            combined_text = ' '.join(speaker_texts[spk_id])[:1000]  # Use first 1000 chars

            # Try to match against known speakers
            best_match = None
            best_score = 0

            for name in self.transcript_speakers:
                if name in used_names:
                    continue

                # Simple fuzzy matching - could be improved
                # Check if name appears in the text
                if name.lower() in combined_text.lower():
                    best_match = name
                    best_score = 100
                    break

                # Fuzzy match
                score = fuzz.partial_ratio(name.lower(), combined_text.lower())
                if score > best_score:
                    best_score = score
                    best_match = name

            if best_match and best_score > 40:  # Threshold
                speaker_mapping[spk_id] = best_match
                used_names.add(best_match)
                logger.info(f"Matched {spk_id} -> {best_match} (score: {best_score})")
            else:
                speaker_mapping[spk_id] = f"Unknown_{spk_id}"
                logger.warning(f"Could not confidently match {spk_id}")

        return speaker_mapping


class EarningsCallAligner:
    """Main class to align earnings call audio with transcript"""

    def __init__(self, fiscal_period: str, hf_token: Optional[str] = None, use_hf: bool = False):
        self.fiscal_period = fiscal_period
        self.hf_token = hf_token

        # Determine file paths
        if use_hf:
            # Download from HuggingFace
            logger.info(f"Using HuggingFace dataset: {HF_DATASET_REPO}")
            self.audio_path, self.transcript_path = download_from_hf(fiscal_period)
        else:
            # Use local files
            self.audio_path = AUDIO_DIR / f"NVDA_{fiscal_period}.mp3"
            self.transcript_path = TRANSCRIPT_DIR / f"NVDA_{fiscal_period}.json"

            # Check if files exist
            if not self.audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")
            if not self.transcript_path.exists():
                raise FileNotFoundError(f"Transcript file not found: {self.transcript_path}")

    def process(self) -> Dict:
        """
        Main processing pipeline
        Returns: Aligned data dictionary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {self.fiscal_period}")
        logger.info(f"{'='*80}")

        # Step 1: Parse transcript
        logger.info("Step 1/6: Parsing transcript...")
        parser = TranscriptParser(self.transcript_path)
        transcript_speakers = parser.extract_speakers()
        qa_char_pos = parser.detect_qa_start()

        # Step 2: Transcribe audio with Whisper
        logger.info("Step 2/6: Transcribing audio with Whisper...")
        audio_processor = AudioProcessor(self.audio_path, self.hf_token)
        whisper_segments = audio_processor.transcribe_with_whisper()

        # Step 3: Speaker diarization
        logger.info("Step 3/6: Performing speaker diarization...")
        diarized_segments = audio_processor.diarize_speakers()

        if not diarized_segments:
            logger.warning("Skipping speaker matching due to diarization failure")
            speaker_mapping = {}
        else:
            # Step 4: Match speakers
            logger.info("Step 4/6: Matching speakers to names...")
            matcher = SpeakerMatcher(diarized_segments, whisper_segments, transcript_speakers)
            speaker_mapping = matcher.match_speakers()

        # Step 5: Detect Q&A start in audio
        logger.info("Step 5/6: Detecting section boundaries...")
        qa_audio_start = self._detect_qa_in_audio(whisper_segments, qa_char_pos)

        # Step 6: Create aligned segments
        logger.info("Step 6/6: Creating aligned segments...")
        aligned_segments = self._create_aligned_segments(
            diarized_segments, whisper_segments, speaker_mapping,
            transcript_speakers, qa_audio_start
        )

        # Compile results
        result = {
            'fiscal_period': self.fiscal_period,
            'audio_duration': audio_processor.duration_seconds,
            'speakers': transcript_speakers,
            'speaker_mapping': speaker_mapping,
            'segments': aligned_segments,
            'sections': {
                'prepared_remarks': {'start': 0, 'end': qa_audio_start or audio_processor.duration_seconds},
                'qa': {'start': qa_audio_start or 0, 'end': audio_processor.duration_seconds}
            }
        }

        return result

    def _detect_qa_in_audio(self, whisper_segments: List[Dict], qa_char_pos: Optional[int]) -> Optional[float]:
        """Detect Q&A start time in audio"""
        if not qa_char_pos:
            # Heuristic: Q&A usually starts around 50-70% into the call
            return None

        # Look for Q&A markers in Whisper transcription
        qa_markers = ['question', 'answer', 'operator', 'first question']

        # Start looking around the middle of the call
        for segment in whisper_segments:
            text_lower = segment['text'].lower()
            if any(marker in text_lower for marker in qa_markers):
                if segment['start'] > 600:  # At least 10 min in
                    logger.info(f"Q&A section detected at {segment['start']:.1f}s")
                    return segment['start']

        return None

    def _create_aligned_segments(self, diarized_segments: List[Dict],
                                whisper_segments: List[Dict],
                                speaker_mapping: Dict[str, str],
                                transcript_speakers: Dict[str, str],
                                qa_start: Optional[float]) -> List[Dict]:
        """Create final aligned segments with all information"""
        aligned = []

        for diar_seg in diarized_segments:
            # Get speaker name
            speaker_id = diar_seg['speaker_id']
            speaker_name = speaker_mapping.get(speaker_id, f"Unknown_{speaker_id}")
            speaker_role = transcript_speakers.get(speaker_name, "Unknown")

            # Get text from Whisper for this time range
            text_parts = []
            for whisper_seg in whisper_segments:
                if (whisper_seg['start'] < diar_seg['end'] and
                    whisper_seg['end'] > diar_seg['start']):
                    text_parts.append(whisper_seg['text'])

            text = ' '.join(text_parts).strip()

            # Determine section
            section = 'prepared_remarks'
            if qa_start and diar_seg['start'] >= qa_start:
                section = 'qa'

            aligned.append({
                'start': diar_seg['start'],
                'end': diar_seg['end'],
                'duration': diar_seg['end'] - diar_seg['start'],
                'speaker': speaker_name,
                'speaker_role': speaker_role,
                'speaker_id': speaker_id,
                'text': text,
                'section': section
            })

        return aligned

    def save_outputs(self, aligned_data: Dict):
        """Save all output formats"""
        fiscal_period = aligned_data['fiscal_period']

        # 1. Save aligned JSON
        logger.info("Saving aligned JSON...")
        json_path = OUTPUT_ALIGNED_DIR / f"NVDA_{fiscal_period}_aligned.json"
        with open(json_path, 'w') as f:
            json.dump(aligned_data, f, indent=2)
        logger.info(f"✓ Saved: {json_path}")

        # 2. Split MP3 into sections
        logger.info("Splitting MP3 into sections...")
        self._split_audio_by_section(aligned_data)

        # 3. Extract speaker segments
        logger.info("Extracting speaker segments...")
        self._extract_speaker_segments(aligned_data)

        logger.info("All outputs saved successfully!")

    def _split_audio_by_section(self, aligned_data: Dict):
        """Split audio into prepared_remarks and qa sections"""
        audio = AudioSegment.from_mp3(str(self.audio_path))
        sections = aligned_data['sections']
        fiscal_period = aligned_data['fiscal_period']

        # Prepared remarks
        pr_start = int(sections['prepared_remarks']['start'] * 1000)
        pr_end = int(sections['prepared_remarks']['end'] * 1000)
        pr_audio = audio[pr_start:pr_end]
        pr_path = OUTPUT_SPLIT_DIR / f"NVDA_{fiscal_period}_prepared_remarks.mp3"
        pr_audio.export(str(pr_path), format="mp3")
        logger.info(f"✓ Saved: {pr_path}")

        # Q&A
        qa_start = int(sections['qa']['start'] * 1000)
        qa_end = int(sections['qa']['end'] * 1000)
        qa_audio = audio[qa_start:qa_end]
        qa_path = OUTPUT_SPLIT_DIR / f"NVDA_{fiscal_period}_qa.mp3"
        qa_audio.export(str(qa_path), format="mp3")
        logger.info(f"✓ Saved: {qa_path}")

    def _extract_speaker_segments(self, aligned_data: Dict):
        """Extract and concatenate all segments for each speaker"""
        audio = AudioSegment.from_mp3(str(self.audio_path))
        fiscal_period = aligned_data['fiscal_period']

        # Group segments by speaker
        speaker_segments = {}
        for segment in aligned_data['segments']:
            speaker = segment['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)

        # Create directory for this fiscal period
        period_dir = OUTPUT_SPEAKER_DIR / fiscal_period
        period_dir.mkdir(parents=True, exist_ok=True)

        # Export each speaker's segments
        for speaker, segments in speaker_segments.items():
            # Concatenate all segments for this speaker
            speaker_audio = AudioSegment.empty()
            for seg in segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                speaker_audio += audio[start_ms:end_ms]

            # Save
            safe_filename = re.sub(r'[^\w\s-]', '', speaker.lower()).replace(' ', '_')
            speaker_path = period_dir / f"{safe_filename}.mp3"
            speaker_audio.export(str(speaker_path), format="mp3")
            logger.info(f"✓ Saved: {speaker_path} ({len(segments)} segments)")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Align earnings call audio with transcripts")
    parser.add_argument("--file", type=str, help="Process specific fiscal period (e.g., Q4_2025)")
    parser.add_argument("--all", action="store_true", help="Process all available files")
    parser.add_argument("--hf-token", type=str, help="HuggingFace API token for pyannote")
    parser.add_argument("--use-hf", action="store_true", help="Load audio/transcripts from HuggingFace dataset")

    args = parser.parse_args()

    # Setup
    setup_directories()

    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        logger.warning("No HuggingFace token provided!")
        logger.warning("Speaker diarization will be skipped.")
        logger.warning("Set HF_TOKEN environment variable or use --hf-token parameter")

    # Determine which files to process
    if args.all:
        if args.use_hf:
            # When using HF, list available periods (hardcoded for now)
            fiscal_periods = ['Q4_2025', 'Q1_2026', 'Q2_2026']
            logger.info(f"Using HuggingFace dataset - processing {len(fiscal_periods)} files")
        else:
            # Find all matching transcript/audio pairs locally
            transcript_files = list(TRANSCRIPT_DIR.glob("NVDA_*.json"))
            fiscal_periods = []
            for tf in transcript_files:
                period = tf.stem.replace('NVDA_', '')
                audio_file = AUDIO_DIR / f"NVDA_{period}.mp3"
                if audio_file.exists():
                    fiscal_periods.append(period)

            if not fiscal_periods:
                logger.error("No matching transcript/audio pairs found!")
                return

            logger.info(f"Found {len(fiscal_periods)} files to process")

    elif args.file:
        fiscal_periods = [args.file]
    else:
        logger.error("Must specify --file or --all")
        return

    # Process each file
    for i, period in enumerate(fiscal_periods, 1):
        try:
            logger.info(f"\n{'#'*80}")
            logger.info(f"File {i}/{len(fiscal_periods)}: {period}")
            logger.info(f"{'#'*80}\n")

            aligner = EarningsCallAligner(period, hf_token=hf_token, use_hf=args.use_hf)
            aligned_data = aligner.process()
            aligner.save_outputs(aligned_data)

            logger.info(f"\n✓ Completed {period}")

        except Exception as e:
            logger.error(f"Failed to process {period}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    logger.info(f"\n{'='*80}")
    logger.info("All processing complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
