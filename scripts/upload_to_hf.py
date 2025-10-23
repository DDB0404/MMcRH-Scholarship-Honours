#!/usr/bin/env python3
"""
Upload NVDA earnings call audio and transcripts to HuggingFace dataset
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

# Configuration
REPO_ID = "baws004/nvda-audio"
DATA_DIR = Path(__file__).parent.parent / "data"

def upload_files():
    """Upload audio and transcript files to HuggingFace"""

    # Login (will use HF_TOKEN from environment or prompt for login)
    print("Logging in to HuggingFace...")
    login(token=os.getenv("HF_TOKEN"))

    api = HfApi()

    # Files to upload
    audio_files = list((DATA_DIR / "earnings-calls" / "NVDA").glob("*.mp3"))
    transcript_files = list((DATA_DIR / "earnings-transcripts" / "NVDA").glob("*.json"))

    print(f"\nFound {len(audio_files)} audio files and {len(transcript_files)} transcripts")
    print(f"Uploading to: {REPO_ID}\n")

    # Upload audio files
    print("Uploading audio files...")
    for audio_file in sorted(audio_files):
        print(f"  • {audio_file.name} ({audio_file.stat().st_size / 1_000_000:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(audio_file),
            path_in_repo=f"audio/{audio_file.name}",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    # Upload transcript files
    print("\nUploading transcript files...")
    for transcript_file in sorted(transcript_files):
        print(f"  • {transcript_file.name} ({transcript_file.stat().st_size / 1000:.1f} KB)")
        api.upload_file(
            path_or_fileobj=str(transcript_file),
            path_in_repo=f"transcripts/{transcript_file.name}",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print(f"\n✅ Upload complete! View at: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    upload_files()
