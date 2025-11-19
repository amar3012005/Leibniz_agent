#!/usr/bin/env python3
"""
Remove Emoji Characters from Leibniz Agent Pipeline Files
=========================================================

This script removes all emoji and problematic Unicode characters from files
in the Leibniz agent pipeline that may cause encoding or processing errors.

Emojis and certain Unicode characters can cause issues with:
- File encoding on Windows systems
- Text processing in various components
- Database storage and retrieval
- API communications
- Logging and debugging output

Usage:
    python remove_emojis.py                    # Process main pipeline files
    python remove_emojis.py --all-files        # Process all text files in directory
    python remove_emojis.py --scan-recursive   # Process all text files recursively
    python remove_emojis.py file1.py file2.py  # Process specific files
    python remove_emojis.py --dry-run          # Show what would be changed
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Set


class EmojiRemover:
    """Handles emoji detection and removal from text files."""

    def __init__(self):
        # Comprehensive emoji regex pattern
        self.emoji_pattern = re.compile(
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
            "\u00a9"  # copyright
            "\u00ae"  # registered
            "\u2122"  # trademark
            "]+",
            flags=re.UNICODE
        )

        # Additional patterns for problematic Unicode characters
        self.problematic_unicode = re.compile(
            r'[\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\ufeff]'
        )

        # Control characters that can cause issues
        self.control_chars = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

    def contains_problematic_chars(self, text: str) -> bool:
        """Check if text contains emoji or problematic Unicode characters."""
        return bool(
            self.emoji_pattern.search(text) or
            self.problematic_unicode.search(text) or
            self.control_chars.search(text)
        )

    def remove_problematic_chars(self, text: str) -> Tuple[str, int]:
        """
        Remove emoji and problematic Unicode characters from text.

        Returns:
            Tuple of (cleaned_text, count_of_removed_chars)
        """
        original_len = len(text)

        # Remove emojis
        text = self.emoji_pattern.sub('', text)
        # Remove other problematic Unicode characters
        text = self.problematic_unicode.sub('', text)
        # Remove control characters
        text = self.control_chars.sub('', text)

        removed_count = original_len - len(text)
        return text, removed_count

    def is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file based on extension."""
        text_extensions = {
            '.py', '.txt', '.md', '.json', '.yaml', '.yml', '.ini', '.cfg', '.conf',
            '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.xml', '.html', '.css'
        }
        return file_path.suffix.lower() in text_extensions


def get_pipeline_files(directory: Path) -> List[Path]:
    """Get list of main pipeline files in the Leibniz agent."""
    pipeline_files = [
        'leibniz_pro.py',
        'leibniz_vad.py',
        'leibniz_stt.py',
        'leibniz_tts.py',
        'leibniz_rag.py',
        'leibniz_intent_parser.py',
        'leibniz_semantic_extractor.py',
        'leibniz_dialogue_manager.py',
        'leibniz_webrtc_app.py',
        'leibniz_webrtc_handler.py',
        'leibniz_webrtc_io.py',
        'leibniz_continuous_vad.py',
        'leibniz_appointment_fsm.py',
        'leibniz_persistent_services.py',
        'leibniz_config.py',
        'leibniz_messages.py',
        'leibniz_semantic_translator.py'
    ]

    found_files = []
    for filename in pipeline_files:
        file_path = directory / filename
        if file_path.exists():
            found_files.append(file_path)

    return found_files


def find_text_files(directory: Path, recursive: bool = False) -> List[Path]:
    """Find all text files in the directory."""
    remover = EmojiRemover()
    text_files = []

    try:
        if recursive:
            for file_path in directory.rglob('*'):
                if file_path.is_file() and remover.is_text_file(file_path):
                    # Skip common directories
                    if not any(part.startswith('.') or part in {'__pycache__', 'node_modules', 'build', 'dist'}
                             for part in file_path.parts):
                        text_files.append(file_path)
        else:
            for file_path in directory.iterdir():
                if file_path.is_file() and remover.is_text_file(file_path):
                    text_files.append(file_path)
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")

    return text_files


def process_file(filepath: Path, create_backup: bool = True, dry_run: bool = False) -> Tuple[int, bool]:
    """
    Process a single file to remove problematic characters.

    Returns:
        Tuple of (characters_removed, file_was_modified)
    """
    remover = EmojiRemover()

    try:
        # Read the file
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            original_content = f.read()

        # Check if file contains problematic characters
        if not remover.contains_problematic_chars(original_content):
            return 0, False

        # Remove problematic characters
        cleaned_content, chars_removed = remover.remove_problematic_chars(original_content)

        if chars_removed == 0:
            return 0, False

        if dry_run:
            print(f"📋 Would clean: {filepath} ({chars_removed} problematic characters)")
            return chars_removed, False

        # Create backup if requested
        if create_backup:
            backup_path = filepath.with_suffix(filepath.suffix + '.emoji_backup')
            try:
                shutil.copy2(filepath, backup_path)
                print(f"📁 Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️  Failed to create backup for {filepath}: {e}")

        # Write cleaned content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"🧹 Cleaned: {filepath} (removed {chars_removed} characters)")
        return chars_removed, True

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return 0, False


def main():
    """Main function to remove problematic characters from files."""
    print("=" * 70)
    print("REMOVE PROBLEMATIC CHARACTERS FROM LEIBNIZ AGENT PIPELINE")
    print("=" * 70)

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Remove emoji and problematic Unicode characters")
    parser.add_argument('files', nargs='*', help='Specific files to process')
    parser.add_argument('--all-files', action='store_true', help='Process all text files in current directory')
    parser.add_argument('--scan-recursive', action='store_true', help='Process all text files recursively')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without modifying files')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backup files')

    args = parser.parse_args()

    # Determine which files to process
    current_dir = Path.cwd()

    if args.files:
        # Specific files provided
        files_to_process = [Path(f) for f in args.files]
        print(f"🎯 Processing {len(files_to_process)} specified files")
    elif args.all_files:
        # All text files in current directory
        files_to_process = find_text_files(current_dir, recursive=False)
        print(f"📂 Processing all {len(files_to_process)} text files in current directory")
    elif args.scan_recursive:
        # All text files recursively
        files_to_process = find_text_files(current_dir, recursive=True)
        print(f"📂 Processing all {len(files_to_process)} text files recursively")
    else:
        # Default: main pipeline files
        files_to_process = get_pipeline_files(current_dir)
        print(f"🔧 Processing {len(files_to_process)} main pipeline files")

    if not files_to_process:
        print("❌ No files found to process!")
        return 1

    # Process files
    total_chars_removed = 0
    files_modified = 0
    create_backup = not args.no_backup

    print(f"\n🧹 Starting character removal...")
    print(f"📁 Backup creation: {'enabled' if create_backup else 'disabled'}")
    print(f"👀 Dry run: {'enabled' if args.dry_run else 'disabled'}\n")

    for filepath in files_to_process:
        if not filepath.exists():
            print(f"⚠️  File not found: {filepath}")
            continue

        chars_removed, was_modified = process_file(filepath, create_backup, args.dry_run)
        total_chars_removed += chars_removed
        if was_modified or (args.dry_run and chars_removed > 0):
            files_modified += 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if args.dry_run:
        print(f"Files that would be modified: {files_modified}")
        print(f"Total characters that would be removed: {total_chars_removed}")
    else:
        print(f"Files processed: {len(files_to_process)}")
        print(f"Files modified: {files_modified}")
        print(f"Total characters removed: {total_chars_removed}")

        if total_chars_removed > 0:
            print("\n✅ SUCCESS: Problematic characters removed!")
            if create_backup:
                print("💡 To restore backups: rename *.emoji_backup files back to original names")
        else:
            print("\n✅ No problematic characters found - all files are clean!")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
