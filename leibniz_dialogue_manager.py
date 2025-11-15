"""
Leibniz Dialogue Manager
Centralized dialogue management using environment variables for Leibniz Agent.

This module provides:
- Environment-based dialogue loading (no JSON database)
- MD5-based cache naming for consistent audio file hashing
- Structured audio archiving with intent-based folders
- Clean APIs for dialogue retrieval and audio playback
"""

import os
import hashlib
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DialogueConfig:
    """Configuration for dialogue management"""
    enable_archiving: bool = False
    archive_base_dir: str = "leibniz_agent/audio/dialogues"
    use_categories: bool = True
    reuse_audio: bool = True

class LeibnizDialogueManager:
    """
    Centralized dialogue manager for Leibniz Agent.

    Loads dialogues from environment variables only,
    manages structured audio archive with intent-based folders.
    """

    def __init__(self, config: Optional[DialogueConfig] = None):
        """Initialize dialogue manager with configuration"""
        self.config = config or DialogueConfig()
        self._dialogues = {}
        self._dialogue_categories = {}
        self._load_dialogues()

    def _load_dialogues(self):
        """Load dialogues from category files and environment variables"""
        # Define category mappings for structured organization
        self._dialogue_categories = {
            # Greetings category
            'greetings': 'greetings',
            'intro': 'greetings',
            'welcome': 'greetings',
            'help': 'greetings',

            # Farewells category
            'farewells': 'farewells',
            'outro': 'farewells',
            'exit': 'farewells',

            # Prompts category
            'prompts': 'prompts',
            'help_offer': 'prompts',
            'continue': 'prompts',
            'clarify': 'prompts',
            'confirmation': 'prompts',

            # Errors category
            'errors': 'errors',
            'error': 'errors',
            'timeout': 'errors',
            'technical': 'errors',
            'no_input': 'errors',
            'general': 'errors',

            # Appointments category
            'appointments': 'appointments',
            'appointment_confirm': 'appointments',
            'date_prompt': 'appointments',
            'time_prompt': 'appointments',
            'email_prompt': 'appointments',
            'name_prompt': 'appointments',

            # Thinking category
            'thinking': 'thinking',
            'thinking_indicator': 'thinking',
        }

        # First, load from category files
        self._load_from_category_files()

        # Then load from direct environment variables (these can override file-based ones)
        self._load_from_environment_variables()

        logger.info(f"✅ Loaded {len(self._dialogues)} dialogues from files and environment variables")

    def _load_from_category_files(self):
        """Load dialogues from category-specific .env files"""
        # Get the directory where this script is located (leibniz_agent/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define category file mappings (relative to script directory)
        category_files = {
            'intro': os.getenv('LEIBNIZ_DIALOGUE_INTRO_FILE', os.path.join(script_dir, 'dialogues', '.env.intro')),
            'fsm': os.getenv('LEIBNIZ_DIALOGUE_FSM_FILE', os.path.join(script_dir, 'dialogues', '.env.fsm')),
            'rag': os.getenv('LEIBNIZ_DIALOGUE_RAG_FILE', os.path.join(script_dir, 'dialogues', '.env.rag')),
            'errors': os.getenv('LEIBNIZ_DIALOGUE_ERRORS_FILE', os.path.join(script_dir, 'dialogues', '.env.errors')),
            'prompts': os.getenv('LEIBNIZ_DIALOGUE_PROMPTS_FILE', os.path.join(script_dir, 'dialogues', '.env.prompts')),
        }

        # Load each category file
        for category, file_path in category_files.items():
            self._load_env_file(file_path, category)

    def _load_env_file(self, file_path: str, category: str):
        """
        Load environment variables from a specific .env file.

        Args:
            file_path: Path to the .env file
            category: Category name for mapping
        """
        try:
            from pathlib import Path

            env_file = Path(file_path)
            if not env_file.exists():
                logger.warning(f"Dialogue file not found: {file_path}")
                return

            # Read and parse the .env file
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Only process LEIBNIZ_DIALOGUE_ prefixed variables
                        if key.startswith('LEIBNIZ_DIALOGUE_'):
                            # Extract the dialogue key and map to internal format
                            internal_key = self._map_file_category_to_internal(key, category)
                            if internal_key:
                                self._dialogues[internal_key] = value

            logger.debug(f"✅ Loaded dialogues from {file_path} ({category} category)")

        except Exception as e:
            logger.warning(f"Failed to load dialogue file {file_path}: {e}")

    def _map_file_category_to_internal(self, env_key: str, file_category: str) -> Optional[str]:
        """
        Map environment variable key from file to internal dialogue key format.

        Args:
            env_key: Environment variable key (e.g., 'LEIBNIZ_DIALOGUE_GREETINGS_INTRO')
            file_category: Category from filename (e.g., 'intro')

        Returns:
            Internal key in category.key format (e.g., 'greetings.intro') or None if invalid
        """
        # Remove LEIBNIZ_DIALOGUE_ prefix
        dialogue_key = env_key.replace('LEIBNIZ_DIALOGUE_', '')

        # Split on underscores to get category and subkey
        parts = dialogue_key.split('_', 1)
        if len(parts) != 2:
            logger.warning(f"Invalid dialogue key format: {env_key}")
            return None

        file_cat, subkey = parts

        # Map file category to internal category
        category_mapping = {
            'intro': 'greetings',  # intro file maps to greetings category
            'fsm': 'prompts',      # fsm file maps to prompts category
            'rag': 'prompts',      # rag file maps to prompts category
            'errors': 'errors',    # errors file maps to errors category
            'prompts': 'prompts',  # prompts file maps to prompts category
        }

        internal_category = category_mapping.get(file_category, file_cat.lower())

        # Return in category.key format
        return f"{internal_category}.{subkey.lower()}"

    def _load_from_environment_variables(self):
        """Load dialogues from direct environment variables (can override file-based ones)"""
        # Format: LEIBNIZ_DIALOGUE_<CATEGORY>_<KEY>=text
        for key, value in os.environ.items():
            if key.startswith('LEIBNIZ_DIALOGUE_'):
                # Extract category and key from environment variable name
                # LEIBNIZ_DIALOGUE_GREETINGS_INTRO → category='greetings', key='intro'
                dialogue_key = key.replace('LEIBNIZ_DIALOGUE_', '').lower()
                parts = dialogue_key.split('_', 1)  # Split on first underscore only

                if len(parts) == 2:
                    category, subkey = parts
                    # Store as category.key format for easy lookup
                    full_key = f"{category}.{subkey}"
                    self._dialogues[full_key] = value
                else:
                    # Fallback for variables without category prefix
                    self._dialogues[dialogue_key] = value

        logger.debug(f"✅ Loaded dialogues from environment variables")

    def get_dialogue(self, category: str, key: str, **kwargs) -> Optional[str]:
        """
        Get dialogue text by category and key.

        Args:
            category: Dialogue category (e.g., 'greetings', 'errors', 'prompts')
            key: Dialogue key within category (e.g., 'intro', 'timeout', 'help')
            **kwargs: Optional formatting parameters

        Returns:
            Dialogue text or None if not found
        """
        full_key = f"{category}.{key}".lower()
        dialogue = self._dialogues.get(full_key)

        if dialogue is None:
            logger.warning(f"Dialogue key '{full_key}' not found")
            return None

        # Format the string with provided kwargs if any
        if kwargs:
            try:
                return dialogue.format(**kwargs)
            except KeyError as e:
                logger.warning(f"Missing format parameter for '{full_key}': {e}")
                return dialogue

        return dialogue

    def get_dialogue_category(self, key: str) -> str:
        """
        Get category for a dialogue key.

        Args:
            key: Dialogue key

        Returns:
            Category string (greetings, farewells, prompts, errors, appointments, thinking)
        """
        # Extract base key (remove hierarchical prefix)
        base_key = key.lower().split('.')[-1]

        # Check explicit category mapping
        category = self._dialogue_categories.get(base_key, 'prompts')  # Default to prompts

        return category

    def get_audio_path(self, category: str, key: str, text: str, emotion: str) -> Optional[str]:
        """
        Get standardized audio file path for dialogue.

        Args:
            category: Dialogue category
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone

        Returns:
            Full path to audio file or None if not found
        """
        if not self.config.enable_archiving:
            return None

        # Generate standardized filename: <dialogue_key>_<md5_hash>.wav
        filename = self._generate_audio_filename(key, text, emotion)
        category_dir = os.path.join(self.config.archive_base_dir, category)

        return os.path.join(category_dir, filename)

    def check_audio_exists(self, category: str, key: str, text: str, emotion: str) -> bool:
        """
        Check if audio file exists for dialogue.

        Args:
            category: Dialogue category
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone

        Returns:
            True if audio file exists
        """
        audio_path = self.get_audio_path(category, key, text, emotion)
        return audio_path is not None and os.path.exists(audio_path)

    def ensure_audio_exists(self, category: str, key: str, text: str, emotion: str, tts_synthesizer) -> Optional[str]:
        """
        Ensure audio exists for dialogue - check archive, synthesize if missing, save to archive.

        Args:
            category: Dialogue category
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone
            tts_synthesizer: TTS synthesizer function/class

        Returns:
            Path to audio file (existing or newly created)
        """
        # Check if audio already exists
        audio_path = self.get_audio_path(category, key, text, emotion)
        if audio_path and os.path.exists(audio_path):
            logger.debug(f"✅ Using archived dialogue audio: {category}.{key}")
            return audio_path

        # Synthesize new audio
        try:
            # Generate standardized filename
            filename = self._generate_audio_filename(key, text, emotion)
            temp_path = os.path.join(self.config.archive_base_dir, "temp", filename)

            # Ensure temp directory exists
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)

            # Synthesize audio
            if hasattr(tts_synthesizer, 'synthesize_to_file'):
                # TTS class method
                success = tts_synthesizer.synthesize_to_file(text, temp_path, emotion=emotion)
            else:
                # TTS function
                success = tts_synthesizer(text, temp_path, emotion=emotion)

            if not success:
                logger.warning(f"Failed to synthesize audio for {category}.{key}")
                return None

            # Move to final location and save metadata
            final_path = self._archive_audio_file(temp_path, category, key, text, emotion)

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            return final_path

        except Exception as e:
            logger.warning(f"Failed to ensure audio exists for {category}.{key}: {e}")
            return None

    def _generate_audio_filename(self, key: str, text: str, emotion: str) -> str:
        """
        Generate standardized audio filename with MD5 hash.

        Args:
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone

        Returns:
            Filename: <dialogue_key>_<md5_hash>.wav
        """
        # Create hash from text + emotion for uniqueness
        hash_input = f"{text}|{emotion}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()

        return f"{key}_{hash_value}.wav"

    def _archive_audio_file(self, temp_path: str, category: str, key: str, text: str, emotion: str) -> Optional[str]:
        """
        Archive audio file to category-based folder structure with metadata.

        Args:
            temp_path: Path to temporary audio file
            category: Dialogue category
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone

        Returns:
            Path to archived file or None if archiving failed
        """
        if not self.config.enable_archiving:
            return temp_path  # Return temp path if not archiving

        try:
            import shutil

            # Create category directory
            category_dir = os.path.join(self.config.archive_base_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            # Generate final filename
            filename = self._generate_audio_filename(key, text, emotion)
            final_path = os.path.join(category_dir, filename)

            # Move file to final location
            shutil.move(temp_path, final_path)

            # Save metadata
            self._save_metadata(category, key, text, emotion, final_path)

            logger.debug(f"📁 Archived dialogue audio: {final_path}")
            return final_path

        except Exception as e:
            logger.warning(f"Failed to archive dialogue audio: {e}")
            return temp_path  # Return temp path as fallback

    def _save_metadata(self, category: str, key: str, text: str, emotion: str, audio_path: str):
        """
        Save JSON metadata alongside audio file.

        Args:
            category: Dialogue category
            key: Dialogue key
            text: Dialogue text
            emotion: Emotion/tone
            audio_path: Path to audio file
        """
        try:
            metadata_path = audio_path.replace('.wav', '_metadata.json')
            metadata = {
                "category": category,
                "key": key,
                "text": text,
                "emotion": emotion,
                "audio_file": os.path.basename(audio_path),
                "created": os.path.getctime(audio_path),
                "file_size": os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to save metadata for {category}.{key}: {e}")

    def get_audio_path_from_cache_name(self, cache_name: str, text: str, emotion: str) -> Optional[str]:
        """
        Get audio path from cache name (for backward compatibility).

        Args:
            cache_name: Legacy cache name
            text: Dialogue text
            emotion: Emotion/tone

        Returns:
            Path to audio file or None
        """
        # Try to infer category from cache_name
        category = self.get_dialogue_category(cache_name)
        return self.get_audio_path(category, cache_name, text, emotion)

    def archive_dialogue_audio(self, cache_name: str, text: str, emotion: str, audio_file: str) -> Optional[str]:
        """
        Archive dialogue audio (legacy method for backward compatibility).

        Args:
            cache_name: Legacy cache name
            text: Dialogue text
            emotion: Emotion/tone
            audio_file: Path to audio file to archive

        Returns:
            Path to archived file or None
        """
        category = self.get_dialogue_category(cache_name)
        return self._archive_audio_file(audio_file, category, cache_name, text, emotion)

    def list_dialogues(self) -> Dict[str, str]:
        """
        List all available dialogues.

        Returns:
            Dictionary of dialogue keys to texts
        """
        return self._dialogues.copy()

    @property
    def dialogues(self) -> Dict[str, str]:
        """Property access to dialogues dictionary"""
        return self._dialogues

# Global singleton instance
_dialogue_manager_instance = None

def get_leibniz_dialogue_manager() -> LeibnizDialogueManager:
    """
    Get singleton instance of Leibniz dialogue manager.

    Returns:
        LeibnizDialogueManager instance
    """
    global _dialogue_manager_instance

    if _dialogue_manager_instance is None:
        # Load configuration from environment
        config = DialogueConfig(
            enable_archiving=os.getenv('LEIBNIZ_ENABLE_DIALOGUE_ARCHIVE', 'true').lower() == 'true',
            archive_base_dir=os.getenv('LEIBNIZ_DIALOGUE_ARCHIVE_DIR', 'leibniz_agent/audio/dialogues'),
            use_categories=os.getenv('LEIBNIZ_DIALOGUE_USE_CATEGORIES', 'true').lower() == 'true',
            reuse_audio=os.getenv('LEIBNIZ_DIALOGUE_REUSE_AUDIO', 'true').lower() == 'true'
        )

        _dialogue_manager_instance = LeibnizDialogueManager(config)
        logger.info("✅ Initialized Leibniz dialogue manager")

    return _dialogue_manager_instance