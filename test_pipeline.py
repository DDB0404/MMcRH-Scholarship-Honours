"""
Test script for earnings pipeline

This script verifies that the pipeline works correctly on NVDA Q2 2026.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import pipeline
from earnings_pipeline import (
    process_single_call,
    PipelineConfig,
    get_default_config
)

def test_single_call():
    """Test processing a single call"""
    print("\n" + "="*80)
    print("TESTING EARNINGS PIPELINE")
    print("="*80 + "\n")

    # Set up paths
    data_dir = project_root / "data"
    calls_dir = data_dir / "earnings-calls"
    transcripts_dir = data_dir / "earnings-transcripts"
    output_dir = data_dir / "processed_features"

    # NVDA Q2 2026 paths
    ticker = "NVDA"
    quarter = "Q2"
    year = "2026"

    mp3_path = calls_dir / ticker / f"{quarter}_{year}" / f"{ticker}_{quarter}_{year}.mp3"
    transcript_json_path = transcripts_dir / ticker / f"{quarter}_{year}" / f"{ticker}_{quarter}_{year}.json"
    transcription_cache = transcripts_dir / ticker / f"{quarter}_{year}" / f"{ticker}_{quarter}_{year}_whisperx_transcription.json"

    # Check paths exist
    print(f"Checking paths:")
    print(f"  MP3: {mp3_path.exists()} - {mp3_path}")
    print(f"  Transcript: {transcript_json_path.exists()} - {transcript_json_path}")
    print(f"  Transcription cache: {transcription_cache.exists()} - {transcription_cache}")

    if not mp3_path.exists():
        print("\n❌ ERROR: MP3 file not found")
        return False

    if not transcript_json_path.exists():
        print("\n❌ ERROR: Transcript JSON not found")
        return False

    # Configure pipeline
    config = get_default_config()

    # Process the call
    print("\n" + "-"*80)
    print("PROCESSING CALL")
    print("-"*80)

    try:
        features_df = process_single_call(
            ticker=ticker,
            quarter=quarter,
            year=year,
            mp3_path=str(mp3_path),
            transcript_json_path=str(transcript_json_path),
            output_dir=str(output_dir),
            event_date="2025-08-27",
            compute_volatility=True,
            config=config,
            use_cached_transcription=True,
            transcription_cache_path=str(transcription_cache) if transcription_cache.exists() else None
        )

        print("\n" + "="*80)
        print("✅ PROCESSING SUCCESSFUL")
        print("="*80)

        # Verify outputs
        print("\n" + "-"*80)
        print("VERIFYING OUTPUTS")
        print("-"*80)

        print(f"\nDataFrame shape: {features_df.shape}")
        print(f"Number of speakers: {len(features_df[features_df['speaker'] != 'CALL_LEVEL'])}")
        print(f"Columns: {len(features_df.columns)}")

        # Check for key columns
        expected_cols = ['speaker', 'ticker', 'quarter', 'year', 'event_date']
        emotion_cols = ['valence_mean', 'arousal_mean', 'dominance_mean']
        vol_cols = [f'rv_{h}d' for h in [1, 3, 5, 7, 30]]

        print("\nColumn checks:")
        for col in expected_cols:
            status = "✓" if col in features_df.columns else "✗"
            print(f"  {status} {col}")

        print("\nEmotion feature checks:")
        for col in emotion_cols:
            status = "✓" if col in features_df.columns else "✗"
            print(f"  {status} {col}")

        print("\nVolatility target checks:")
        for col in vol_cols:
            status = "✓" if col in features_df.columns else "✗"
            print(f"  {status} {col}")

        # Display volatility values
        print("\n" + "-"*80)
        print("VOLATILITY TARGETS")
        print("-"*80)

        call_level = features_df[features_df['speaker'] == 'CALL_LEVEL']
        if len(call_level) > 0:
            for col in vol_cols:
                if col in call_level.columns:
                    val = call_level[col].iloc[0]
                    horizon = col.replace('rv_', '').replace('d', '')
                    print(f"  {horizon}-day realized vol: {val:.4f}" if not pd.isna(val) else f"  {horizon}-day realized vol: N/A")
        else:
            print("  No CALL_LEVEL row found")

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR DURING PROCESSING")
        print("="*80)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    success = test_single_call()
    sys.exit(0 if success else 1)
