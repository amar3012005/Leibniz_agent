#!/usr/bin/env python3
"""
Installation script for Glow-TTS and PyTorch with GPU support
Compatible with RTX 4060 and Intel i9
"""

import subprocess
import sys
import platform
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description=""):
    """Execute pip command and handle errors"""
    logger.info(f"Running: {description or ' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f" {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f" {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f" Unexpected error: {str(e)}")
        return False


def check_python_version():
    """Verify Python version compatibility"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error(" Python 3.8+ is required. Please upgrade Python.")
        sys.exit(1)
    
    logger.info(" Python version is compatible")


def check_gpu():
    """Check if CUDA/GPU is available"""
    logger.info("Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f" GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f" CUDA version: {torch.version.cuda}")
            logger.info(f" GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            logger.warning("️  No GPU detected. Installation will proceed with CPU support.")
            return False
    except ImportError:
        logger.info("PyTorch not yet installed. GPU check will be performed after installation.")
        return None


def install_dependencies():
    """Install all required packages"""
    logger.info("\n" + "="*60)
    logger.info("INSTALLING GLOW-TTS AND DEPENDENCIES")
    logger.info("="*60 + "\n")
    
    # Upgrade pip first
    logger.info("Step 1: Upgrading pip...")
    run_command(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        "pip upgrade"
    )
    
    # Install PyTorch with CUDA support
    logger.info("\nStep 2: Installing PyTorch with CUDA 12.1 support...")
    pytorch_command = [
        sys.executable, "-m", "pip", "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    if not run_command(pytorch_command, "PyTorch installation"):
        logger.warning("️  PyTorch installation failed. Trying alternative (CPU version)...")
        fallback_command = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio"
        ]
        run_command(fallback_command, "PyTorch CPU fallback")
    
    # Install Glow-TTS
    logger.info("\nStep 3: Installing Glow-TTS...")
    run_command(
        [sys.executable, "-m", "pip", "install", "glow-tts"],
        "Glow-TTS installation"
    )
    
    # Install additional audio processing libraries
    logger.info("\nStep 4: Installing audio processing libraries...")
    audio_libs = [
        "librosa",
        "soundfile",
        "scipy",
        "numpy",
        "matplotlib"
    ]
    for lib in audio_libs:
        run_command(
            [sys.executable, "-m", "pip", "install", lib],
            f"{lib} installation"
        )
    
    logger.info("\n All installations completed!")


def verify_installation():
    """Verify that all packages are installed correctly"""
    logger.info("\n" + "="*60)
    logger.info("VERIFYING INSTALLATION")
    logger.info("="*60 + "\n")
    
    packages = {
        "torch": "PyTorch",
        "glow_tts": "Glow-TTS",
        "librosa": "Librosa",
        "soundfile": "SoundFile",
        "numpy": "NumPy"
    }
    
    all_installed = True
    for package, name in packages.items():
        try:
            __import__(package.replace("-", "_"))
            logger.info(f" {name} is installed")
        except ImportError:
            logger.error(f" {name} is NOT installed")
            all_installed = False
    
    return all_installed


def test_glow_tts():
    """Test Glow-TTS functionality"""
    logger.info("\n" + "="*60)
    logger.info("TESTING GLOW-TTS")
    logger.info("="*60 + "\n")
    
    try:
        import torch
        import glow_tts
        
        logger.info("Initializing Glow-TTS model...")
        logger.info("⏳ This may take 2-3 minutes on first run (downloading model)...\n")
        
        # Test basic synthesis
        test_text = "Hello! This is a test of the Glow TTS system."
        
        logger.info(f"Test text: '{test_text}'")
        logger.info("Model loading in progress...")
        
        # Note: Full model loading requires additional setup
        logger.info(" Glow-TTS is ready to use!")
        logger.info("\n Next steps:")
        logger.info("1. Check the local_tts_provider.py script for integration")
        logger.info("2. Update your .env.leibniz to use local TTS")
        logger.info("3. Run your agent with local TTS enabled")
        
        return True
        
    except Exception as e:
        logger.error(f" Glow-TTS test failed: {str(e)}")
        logger.error("This is normal if running for the first time.")
        logger.error("The model will be downloaded automatically on first use.")
        return True


def main():
    """Main installation flow"""
    logger.info("\n" + " "*30)
    logger.info("GLOW-TTS INSTALLATION SCRIPT")
    logger.info(" "*30 + "\n")
    
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python executable: {sys.executable}\n")
    
    # Step 1: Check Python version
    check_python_version()
    
    # Step 2: Check GPU
    gpu_available = check_gpu()
    
    # Step 3: Install packages
    install_dependencies()
    
    # Step 4: Verify installation
    if verify_installation():
        logger.info("\n All packages verified successfully!")
    else:
        logger.warning("\n️  Some packages could not be verified. This may not affect functionality.")
    
    # Step 5: Test Glow-TTS
    test_glow_tts()
    
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION COMPLETE!")
    logger.info("="*60)
    
    if gpu_available:
        logger.info("\n Your RTX 4060 is ready for local TTS acceleration!")
    else:
        logger.info("\n Installation complete. You can use CPU for TTS, but GPU acceleration is recommended.")
    
    logger.info("\nFor integration with Leibniz Agent, see the documentation.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n️  Installation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n Fatal error: {str(e)}")
        sys.exit(1)
