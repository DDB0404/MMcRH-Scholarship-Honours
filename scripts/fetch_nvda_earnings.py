#!/usr/bin/env python3
"""
NVDA Earnings Call Data Fetcher
Fetches earnings call transcripts and audio recordings from Seeking Alpha
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, TimeoutError as PlaywrightTimeout


# Configuration
TRANSCRIPT_DIR = Path("/Users/dominicbyrne/MMcRH-Scholarship-Honours/data/earnings-transcripts/NVDA")
AUDIO_DIR = Path("/Users/dominicbyrne/MMcRH-Scholarship-Honours/data/earnings-calls/NVDA")
SEEKING_ALPHA_BASE = "https://seekingalpha.com"
NVDA_EARNINGS_SEARCH = f"{SEEKING_ALPHA_BASE}/symbol/NVDA/earnings/transcripts"

# Hardcoded list of NVDA earnings call URLs
NVDA_EARNINGS_URLS = [
    "https://seekingalpha.com/article/4762511-nvidia-corporation-nvda-q4-2025-earnings-call-transcript",
    "https://seekingalpha.com/article/4817296-nvidia-corporation-nvda-q2-2026-earnings-call-transcript",
    "https://seekingalpha.com/article/4790673-nvidia-corporation-nvda-q1-2026-earnings-call-transcript",
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directories ready: {TRANSCRIPT_DIR} and {AUDIO_DIR}")


def extract_quarter_info(title: str) -> Dict[str, str]:
    """
    Extract quarter and year information from title
    Example: "NVIDIA Corporation (NVDA) Q4 2025 Earnings Call Transcript"
    Returns: {"quarter": "Q4", "year": "2025", "fiscal_period": "Q4_2025"}
    """
    match = re.search(r'Q(\d)\s+(\d{4})', title)
    if match:
        quarter = f"Q{match.group(1)}"
        year = match.group(2)
        return {
            "quarter": quarter,
            "year": year,
            "fiscal_period": f"{quarter}_{year}"
        }
    return {"quarter": "Unknown", "year": "Unknown", "fiscal_period": "Unknown"}


def discover_earnings_call_urls(page: Page) -> List[Dict[str, str]]:
    """
    Navigate to NVDA earnings transcripts page and discover all earnings call URLs
    Returns list of dicts with 'url' and 'title'
    """
    logger.info("Discovering NVDA earnings call URLs...")

    try:
        page.goto(NVDA_EARNINGS_SEARCH, wait_until="networkidle", timeout=30000)
        time.sleep(3)  # Allow dynamic content to load

        # Get page content and parse
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')

        earnings_calls = []

        # Find all earnings call transcript links
        # Seeking Alpha typically uses article links with specific patterns
        links = soup.find_all('a', href=re.compile(r'/article/\d+-.*earnings-call-transcript'))

        for link in links:
            href = link.get('href')
            title = link.get_text(strip=True)

            # Skip if already processed
            if not href or not title:
                continue

            full_url = urljoin(SEEKING_ALPHA_BASE, href)

            # Check if it's an NVDA earnings call
            if 'nvidia' in title.lower() or 'nvda' in title.lower():
                earnings_calls.append({
                    'url': full_url,
                    'title': title
                })

        # Remove duplicates
        seen = set()
        unique_calls = []
        for call in earnings_calls:
            if call['url'] not in seen:
                seen.add(call['url'])
                unique_calls.append(call)

        logger.info(f"Found {len(unique_calls)} unique earnings call URLs")
        return unique_calls

    except PlaywrightTimeout:
        logger.error("Timeout while loading earnings transcripts page")
        return []
    except Exception as e:
        logger.error(f"Error discovering URLs: {e}")
        return []


def extract_transcript_data(page: Page, url: str, title: str) -> Optional[Dict]:
    """
    Extract transcript text and metadata from an earnings call page
    """
    logger.info(f"Extracting transcript from: {title}")

    try:
        # Try loading with just domcontentloaded instead of networkidle
        # Seeking Alpha has lots of async content that might prevent networkidle
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)  # Give more time for dynamic content to load

        # Try to handle cookie consent banner if present
        try:
            # Common cookie banner button texts
            cookie_button = page.get_by_role("button", name=re.compile(r"(accept|agree|consent|continue)", re.IGNORECASE))
            if cookie_button.is_visible(timeout=2000):
                cookie_button.click()
                logger.info("Clicked cookie consent button")
                time.sleep(2)
        except Exception:
            pass  # No cookie banner or already dismissed

        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')

        # Extract the actual title from the page
        title_elem = soup.find('h1') or soup.find('title')
        if title_elem:
            actual_title = title_elem.get_text(strip=True)
            logger.info(f"Page title: {actual_title}")
        else:
            actual_title = title

        # Extract article metadata
        quarter_info = extract_quarter_info(actual_title)

        # Find the transcript content
        # Seeking Alpha has different possible content containers
        transcript_div = None

        # Try various selectors that Seeking Alpha uses
        possible_selectors = [
            ('div', {'data-test-id': 'content-container'}),
            ('div', {'id': 'article-body'}),
            ('div', {'data-test-id': 'article-content'}),
            ('article', {}),
            ('div', {'class': re.compile(r'(paywall-unlocked|article-content|transcript-content)', re.IGNORECASE)}),
        ]

        for tag, attrs in possible_selectors:
            if attrs:
                transcript_div = soup.find(tag, attrs)
            else:
                transcript_div = soup.find(tag)
            if transcript_div:
                break

        if not transcript_div:
            logger.warning(f"Could not find transcript content container for {title}")
            # Try to get any main content as fallback
            transcript_div = soup.find('main') or soup.find('body')

        # Extract text content
        transcript_text = transcript_div.get_text(separator='\n', strip=True) if transcript_div else "No content found"

        # Check if we're hitting a paywall
        if len(transcript_text) < 500 or 'create a free' in transcript_text.lower() or 'sign up' in transcript_text.lower():
            logger.warning("Possible paywall or login requirement detected!")
            logger.warning(f"Content preview: {transcript_text[:200]}")
            logger.info("Tip: Seeking Alpha may require a free account. Try visiting the URL in the browser window that opened.")

        # Try to find publish date
        date_elem = soup.find('time') or soup.find('span', class_=re.compile(r'date'))
        publish_date = date_elem.get('datetime') or date_elem.get_text(strip=True) if date_elem else "Unknown"

        # Extract speakers (look for common patterns)
        speakers = []
        speaker_pattern = re.compile(r'^([A-Z][a-z]+ [A-Z][a-z]+)', re.MULTILINE)
        potential_speakers = speaker_pattern.findall(transcript_text[:5000])  # Check first part
        speakers = list(set(potential_speakers))[:20]  # Limit to 20 unique speakers

        transcript_data = {
            "symbol": "NVDA",
            "title": actual_title,
            "quarter": quarter_info["quarter"],
            "year": quarter_info["year"],
            "fiscal_period": quarter_info["fiscal_period"],
            "publish_date": publish_date,
            "url": url,
            "speakers": speakers,
            "transcript": transcript_text
        }

        return transcript_data

    except PlaywrightTimeout:
        logger.error(f"Timeout while loading {url}")
        return None
    except Exception as e:
        logger.error(f"Error extracting transcript: {e}")
        return None


def download_audio_via_player(page: Page, fiscal_period: str) -> bool:
    """
    Download audio by clicking the Play button and then the download icon
    This follows the workflow: Click 'Play Earnings Call' -> Click download icon in player
    """
    try:
        logger.info("Looking for 'Play Earnings Call' button...")

        # Wait a bit for the page to fully load
        time.sleep(2)

        # Try to find and click the "Play Earnings Call" button
        # It might be a button or a link with various possible texts
        play_button_patterns = [
            "Play Earnings Call",
            "Play Audio",
            "Listen",
            "play"
        ]

        play_button = None
        for pattern in play_button_patterns:
            try:
                # Try as button
                play_button = page.get_by_role("button", name=re.compile(pattern, re.IGNORECASE))
                if play_button.count() > 0:
                    # Filter out "Go to player" button - we want the actual play button
                    for i in range(play_button.count()):
                        btn_text = play_button.nth(i).inner_text()
                        if 'go to' not in btn_text.lower():
                            play_button = play_button.nth(i)
                            logger.info(f"Found play button: '{btn_text}'")
                            break
                    if play_button and play_button.count() == 1:
                        break

                # Try as link
                play_button = page.get_by_role("link", name=re.compile(pattern, re.IGNORECASE))
                if play_button.count() > 0:
                    logger.info(f"Found play link with text matching '{pattern}'")
                    break
            except Exception as e:
                logger.debug(f"Pattern '{pattern}' not found: {e}")
                continue

        if not play_button or play_button.count() == 0:
            logger.warning("Could not find 'Play Earnings Call' button")
            logger.info("Trying alternative: looking for play icon or audio element...")

            # Try finding by SVG/icon
            try:
                play_icon = page.locator('svg[data-icon="play"], .play-icon, button[aria-label*="play" i]').first
                if play_icon.count() > 0:
                    play_button = play_icon
                    logger.info("Found play button via icon")
            except Exception:
                pass

        if not play_button or play_button.count() == 0:
            logger.warning("Could not find any play button after all attempts")
            return False

        # Scroll button into view and click
        try:
            play_button.scroll_into_view_if_needed()
            time.sleep(1)
            play_button.click(force=True)  # Force click in case of overlays
            logger.info("Clicked play button, waiting for audio player...")
            time.sleep(3)
        except Exception as e:
            logger.error(f"Failed to click play button: {e}")
            return False

        # Look for download button/icon in the audio player
        # The download button is typically in the bottom right of the player
        logger.info("Looking for download button in audio player...")

        # Try to find download button/link with multiple strategies
        download_button = None
        strategies = [
            ('aria-label/title', '[aria-label*="download" i], [title*="download" i]'),
            ('class name', '[class*="download" i]'),
            ('icon classes', '.fa-download, .icon-download, [data-icon="download"]'),
            ('svg download', 'svg[data-icon="download"], svg.download-icon'),
            ('button with download', 'button:has-text("download")'),
            ('a with download', 'a[download], a:has-text("download")')
        ]

        for strategy_name, selector in strategies:
            try:
                candidate = page.locator(selector).first
                if candidate.count() > 0:
                    download_button = candidate
                    logger.info(f"Found download button via {strategy_name}")
                    break
            except Exception as e:
                logger.debug(f"Strategy '{strategy_name}' failed: {e}")
                continue

        if download_button and download_button.count() > 0:
            logger.info("Attempting to click download button...")

            try:
                # Scroll into view first
                download_button.scroll_into_view_if_needed()
                time.sleep(1)

                # Set up download expectation
                with page.expect_download(timeout=30000) as download_info:
                    download_button.click(force=True)

                download = download_info.value
                logger.info(f"Download started: {download.suggested_filename}")

                # Save to our audio directory
                audio_path = AUDIO_DIR / f"NVDA_{fiscal_period}.mp3"
                download.save_as(audio_path)
                logger.info(f"Audio downloaded successfully: {audio_path.name}")
                return True
            except Exception as e:
                logger.error(f"Error during download: {e}")
                return False
        else:
            logger.warning("Could not find download button in audio player")
            logger.info("Note: You may need to manually download the audio file")
            return False

    except Exception as e:
        logger.error(f"Error downloading audio via player: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def download_audio(audio_url: str, output_path: Path) -> bool:
    """
    Download audio file from URL
    """
    try:
        logger.info(f"Downloading audio to {output_path.name}...")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(audio_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Audio downloaded successfully: {output_path.name}")
        return True

    except requests.RequestException as e:
        logger.error(f"Failed to download audio: {e}")
        return False


def save_transcript_json(transcript_data: Dict, output_path: Path) -> bool:
    """
    Save transcript data as JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Transcript saved: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to save transcript: {e}")
        return False


def process_earnings_call(page: Page, call_info: Dict, dry_run: bool = False) -> Dict[str, bool]:
    """
    Process a single earnings call: extract transcript and download audio
    Returns dict with success status for transcript and audio
    """
    url = call_info['url']
    title = call_info['title']

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing: {title}")
    logger.info(f"URL: {url}")

    quarter_info = extract_quarter_info(title)
    fiscal_period = quarter_info['fiscal_period']

    # Define output paths
    transcript_path = TRANSCRIPT_DIR / f"NVDA_{fiscal_period}.json"
    audio_path = AUDIO_DIR / f"NVDA_{fiscal_period}.mp3"

    results = {"transcript": False, "audio": False}

    # Check if files already exist
    if transcript_path.exists():
        logger.info(f"Transcript already exists: {transcript_path.name}")
        results["transcript"] = True
    else:
        # Extract and save transcript
        transcript_data = extract_transcript_data(page, url, title)
        if transcript_data and not dry_run:
            results["transcript"] = save_transcript_json(transcript_data, transcript_path)
        elif dry_run:
            logger.info(f"[DRY RUN] Would save transcript to: {transcript_path.name}")
            results["transcript"] = True

    if audio_path.exists():
        logger.info(f"Audio already exists: {audio_path.name}")
        results["audio"] = True
    else:
        # Download audio by clicking Play button and download icon
        if not dry_run:
            results["audio"] = download_audio_via_player(page, fiscal_period)
        else:
            logger.info(f"[DRY RUN] Would click Play and download audio to: {audio_path.name}")
            results["audio"] = True

    return results


def main(dry_run: bool = False, limit: Optional[int] = None, wait_for_login: bool = False):
    """
    Main function to fetch all NVDA earnings calls

    Args:
        dry_run: If True, don't actually download files
        limit: Maximum number of calls to process (None for all)
        wait_for_login: If True, pause before starting to allow manual login
    """
    setup_directories()

    logger.info("Starting NVDA earnings call data fetcher")
    if dry_run:
        logger.info("DRY RUN MODE - No files will be downloaded")

    with sync_playwright() as p:
        # Launch browser (headless=False can help bypass some anti-bot measures)
        browser = p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        # Hide automation
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        # If wait_for_login, navigate to Seeking Alpha and wait
        if wait_for_login:
            logger.info("\n" + "="*80)
            logger.info("MANUAL LOGIN MODE")
            logger.info("="*80)
            logger.info("Opening Seeking Alpha login page...")
            page.goto("https://seekingalpha.com/login")
            logger.info("\nPlease log in to Seeking Alpha in the browser window.")
            logger.info("You have 60 seconds to complete the login...")
            logger.info("="*80 + "\n")

            # Wait 60 seconds for user to log in
            for i in range(60, 0, -10):
                logger.info(f"Waiting for login... {i} seconds remaining")
                time.sleep(10)

            logger.info("Continuing with data fetching...")

        try:
            # Use hardcoded list of earnings call URLs
            # Convert URL list to format expected by process_earnings_call
            earnings_calls = []
            for url in NVDA_EARNINGS_URLS:
                # Extract title from URL for now, will be updated when page loads
                article_id = url.split('/')[-1].replace('-', ' ').title()
                earnings_calls.append({
                    'url': url,
                    'title': article_id  # Placeholder, will extract from page
                })

            logger.info(f"Processing {len(earnings_calls)} hardcoded NVDA earnings call URLs")

            if not earnings_calls:
                logger.warning("No earnings calls found!")
                return

            # Apply limit if specified
            if limit:
                earnings_calls = earnings_calls[:limit]
                logger.info(f"Limited to {limit} earnings calls")

            # Process each earnings call
            stats = {"transcript_success": 0, "audio_success": 0, "total": len(earnings_calls)}

            for i, call_info in enumerate(earnings_calls, 1):
                logger.info(f"\n[{i}/{stats['total']}]")

                results = process_earnings_call(page, call_info, dry_run=dry_run)

                if results["transcript"]:
                    stats["transcript_success"] += 1
                if results["audio"]:
                    stats["audio_success"] += 1

                # Rate limiting
                if i < len(earnings_calls):
                    time.sleep(2)  # Wait between requests

            # Summary
            logger.info(f"\n{'='*80}")
            logger.info("SUMMARY")
            logger.info(f"Total earnings calls: {stats['total']}")
            logger.info(f"Transcripts saved: {stats['transcript_success']}/{stats['total']}")
            logger.info(f"Audio files downloaded: {stats['audio_success']}/{stats['total']}")
            logger.info(f"{'='*80}")

        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            browser.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch NVDA earnings call data from Seeking Alpha")
    parser.add_argument("--dry-run", action="store_true", help="Preview what would be downloaded without actually downloading")
    parser.add_argument("--limit", type=int, help="Limit number of earnings calls to process")
    parser.add_argument("--wait-for-login", action="store_true", help="Open login page and wait for manual login before fetching")

    args = parser.parse_args()

    main(dry_run=args.dry_run, limit=args.limit, wait_for_login=args.wait_for_login)
