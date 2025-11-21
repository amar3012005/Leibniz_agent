#!/usr/bin/env python3
"""
Fix Critical Import Issues in Leibniz Pro
==========================================

This script fixes the most critical import issues and emoji problems
in the core Leibniz Pro files needed for WebRTC functionality.
"""

import os
import re
from pathlib import Path


def fix_import_paths(content: str) -> str:
    """Fix import paths to use proper module prefixes."""
    # Fix Leibniz component imports
    replacements = [
        ('from leibniz_config import', 'from .leibniz_config import'),
        ('from leibniz_messages import', 'from .leibniz_messages import'),
        ('from leibniz_stt import', 'from .leibniz_stt import'),
        ('from leibniz_tts import', 'from .leibniz_tts import'),
        ('from leibniz_intent_parser import', 'from .leibniz_intent_parser import'),
        ('from leibniz_rag import', 'from .leibniz_rag import'),
        ('from leibniz_appointment_fsm import', 'from .leibniz_appointment_fsm import'),
        ('from leibniz_persistent_services import', 'from .leibniz_persistent_services import'),
        ('from leibniz_semantic_translator import', 'from .leibniz_semantic_translator import'),
        ('from leibniz_vad import', 'from .leibniz_vad import'),
        ('from leibniz_continuous_vad import', 'from .leibniz_continuous_vad import'),
        ('from leibniz_dialogue_manager import', 'from .leibniz_dialogue_manager import'),
        ('from leibniz_agent.leibniz_pro import', 'from .leibniz_pro import'),
    ]

    for old, new in replacements:
        content = content.replace(old, new)

    return content


def remove_emojis_from_content(content: str) -> str:
    """Remove emoji characters from content."""
    # Simple emoji removal - remove common emoji Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001f926-\U0001f937"  # gestures
        "\U00010000-\U0010ffff"  # other unicode
        "\u2640-\u2642"  # gender symbols
        "\u2600-\u2B55"  # misc symbols
        "\u200d"  # zero width joiner
        "\u23cf"  # eject symbol
        "\u23e9"  # fast forward
        "\u231a"  # watch
        "\ufe0f"  # variation selector
        "\u3030"  # wavy dash
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', content)


def process_critical_files():
    """Process the most critical files for WebRTC functionality."""
    # Check if we're in the leibniz_agent directory or parent directory
    base_dirs = ["", "leibniz_agent"]

    critical_files = [
        "leibniz_pro.py",
        "leibniz_config.py",
        "leibniz_messages.py",
        "leibniz_stt.py",
        "leibniz_tts.py",
        "leibniz_vad.py",
        "leibniz_rag.py",
        "leibniz_webrtc_io.py",
        "__init__.py"
    ]

    print("Fixing critical files for WebRTC integration...")

    for base_dir in base_dirs:
        for filename in critical_files:
            filepath = os.path.join(base_dir, filename) if base_dir else filename
            if os.path.exists(filepath):
                try:
                    # Read file
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()

                    # Fix import paths
                    content = fix_import_paths(content)

                    # Remove emojis
                    content = remove_emojis_from_content(content)

                    # Write back
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    print(f"  [OK] Fixed {filepath}")

                except Exception as e:
                    print(f"  [ERROR] Failed to process {filepath}: {e}")
            else:
                print(f"  [SKIP] File not found: {filepath}")

    print("Critical file fixes completed!")


if __name__ == "__main__":
    process_critical_files()
