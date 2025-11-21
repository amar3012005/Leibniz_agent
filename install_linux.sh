#!/bin/bash
# Leibniz Agent Linux Installation Script
# This script sets up the Leibniz agent for Ubuntu/Debian Linux systems

set -e  # Exit on any error

echo "🚀 Leibniz Agent Linux Setup"
echo "=============================="

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script is designed for Linux systems only"
    exit 1
fi

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install system dependencies
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    portaudio19-dev \
    python3-dev \
    build-essential \
    libsndfile1 \
    libsndfile1-dev \
    alsa-utils \
    pulseaudio \
    pulseaudio-utils

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv leibniz_env
source leibniz_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Optional: Install TTS-specific dependencies
echo "🎤 Installing TTS dependencies..."
pip install -r services/tts/requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p audio
mkdir -p logs
mkdir -p data

# Set permissions for audio access
echo "🔐 Setting audio permissions..."
sudo usermod -a -G audio $USER
sudo usermod -a -G pulse-access $USER

# Create desktop shortcut (optional)
echo "🖥️  Creating desktop shortcut..."
cat > ~/Desktop/leibniz-agent.desktop << EOF
[Desktop Entry]
Version=1.0
Name=Leibniz Agent
Comment=Leibniz University Assistant
Exec=bash -c "cd $(pwd) && source leibniz_env/bin/activate && python leibniz_fastrtc_server.py"
Icon=$(pwd)/icon.png
Terminal=true
Categories=Education;Utility;
EOF

chmod +x ~/Desktop/leibniz-agent.desktop

echo "✅ Installation complete!"
echo ""
echo "🎯 To run the agent:"
echo "   cd $(pwd)"
echo "   source leibniz_env/bin/activate"
echo "   python leibniz_fastrtc_server.py"
echo ""
echo "🌐 Then open: http://localhost:7860"
echo ""
echo "📝 Note: You may need to log out and back in for audio permissions to take effect."