#!/bin/bash

# Setup script for audio-transcript alignment
# This script helps install dependencies and configure HuggingFace access

set -e  # Exit on error

echo "=================================="
echo "Audio Alignment Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running from correct directory
if [ ! -f "scripts/align_audio_transcript.py" ]; then
    echo -e "${RED}Error: Must run from MMcRH-Scholarship-Honours directory${NC}"
    echo "Usage: bash scripts/setup_audio_alignment.sh"
    exit 1
fi

echo -e "${YELLOW}Step 1: Checking system dependencies...${NC}"

# Check for ffmpeg
if command -v ffmpeg &> /dev/null; then
    echo -e "${GREEN}✓ ffmpeg is installed${NC}"
else
    echo -e "${YELLOW}✗ ffmpeg not found${NC}"
    echo "Installing ffmpeg..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo -e "${RED}Please install Homebrew first: https://brew.sh${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo -e "${RED}Unsupported OS. Please install ffmpeg manually.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ ffmpeg installed${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Installing Python dependencies...${NC}"

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Install requirements
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo -e "${GREEN}✓ Python dependencies installed${NC}"

echo ""
echo -e "${YELLOW}Step 3: HuggingFace setup${NC}"
echo ""
echo "Speaker diarization requires a HuggingFace account and token."
echo ""
echo "Steps:"
echo "1. Create account at: https://huggingface.co/join"
echo "2. Get access token at: https://huggingface.co/settings/tokens"
echo "3. Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
echo ""
read -p "Do you have a HuggingFace token? (y/n): " has_token

if [ "$has_token" == "y" ] || [ "$has_token" == "Y" ]; then
    read -p "Enter your HuggingFace token: " hf_token

    # Add to .bashrc or .zshrc
    SHELL_RC="$HOME/.bashrc"
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_RC="$HOME/.zshrc"
    fi

    # Check if already exists
    if grep -q "export HF_TOKEN=" "$SHELL_RC"; then
        echo "Updating existing HF_TOKEN in $SHELL_RC..."
        sed -i.bak "s/export HF_TOKEN=.*/export HF_TOKEN=\"$hf_token\"/" "$SHELL_RC"
    else
        echo "Adding HF_TOKEN to $SHELL_RC..."
        echo "" >> "$SHELL_RC"
        echo "# HuggingFace token for pyannote.audio" >> "$SHELL_RC"
        echo "export HF_TOKEN=\"$hf_token\"" >> "$SHELL_RC"
    fi

    # Export for current session
    export HF_TOKEN="$hf_token"

    echo -e "${GREEN}✓ HF_TOKEN configured${NC}"
    echo "Token saved to $SHELL_RC"
else
    echo ""
    echo -e "${YELLOW}To set up later, run:${NC}"
    echo "export HF_TOKEN=\"your_token_here\""
    echo ""
    echo -e "${YELLOW}Or add to $SHELL_RC:${NC}"
    echo "echo 'export HF_TOKEN=\"your_token_here\"' >> $SHELL_RC"
fi

echo ""
echo -e "${YELLOW}Step 4: Creating output directories...${NC}"

mkdir -p data/aligned-transcripts/NVDA
mkdir -p data/earnings-calls-split/NVDA
mkdir -p data/speaker-segments/NVDA

echo -e "${GREEN}✓ Output directories created${NC}"

echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "You can now run the alignment script:"
echo ""
echo "  # Process a single call"
echo "  python scripts/align_audio_transcript.py --file Q4_2025"
echo ""
echo "  # Process all calls"
echo "  python scripts/align_audio_transcript.py --all"
echo ""
echo "For detailed usage, see: scripts/AUDIO_ALIGNMENT_GUIDE.md"
echo ""

# Test import
echo "Testing Python imports..."
python -c "import faster_whisper; import pyannote.audio; import pydub; print('✓ All imports successful')" 2>/dev/null && echo -e "${GREEN}✓ Python environment ready${NC}" || echo -e "${YELLOW}⚠ Some imports failed - check dependencies${NC}"

echo ""
