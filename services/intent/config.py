"""
Configuration for Intent Classification Microservice

Extracted from leibniz_intent_parser.py for microservice deployment.

Reference:
    leibniz_agent/leibniz_intent_parser.py (lines 48-86) - Original configuration
    leibniz_agent/services/stt_vad/config.py - Configuration pattern
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IntentConfig:
    """
    Configuration for Intent Classification service.
    
    Attributes:
        gemini_api_key: Gemini API key for LLM fallback (required)
        gemini_model: Gemini model name (default: gemini-2.0-flash-lite)
        confidence_threshold: Fast route confidence threshold (default: 0.8)
        gemini_timeout: Gemini API timeout in seconds (default: 5.0)
        enable_context_extraction: Enable context extraction (default: True)
        log_classifications: Log all classifications (default: True)
        minimal_output: Strip non-essential fields (default: False)
        fast_route_target: Target percentage for fast route (default: 0.8)
    """
    
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash-lite"
    confidence_threshold: float = 0.8
    gemini_timeout: float = 5.0
    enable_context_extraction: bool = True
    log_classifications: bool = True
    minimal_output: bool = False
    fast_route_target: float = 0.8
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        # Validate confidence threshold
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        
        # Validate Gemini timeout
        if self.gemini_timeout <= 0:
            raise ValueError(
                f"gemini_timeout must be positive, got {self.gemini_timeout}"
            )
        
        # Validate fast route target
        if not 0.0 <= self.fast_route_target <= 1.0:
            raise ValueError(
                f"fast_route_target must be between 0.0 and 1.0, got {self.fast_route_target}"
            )
        
        # Warn if Gemini API key is missing
        if not self.gemini_api_key:
            logger.warning(
                "️ GEMINI_API_KEY not set. Gemini fallback will be disabled. "
                "Only fast pattern matching will be available."
            )
        
        # Log configuration if enabled
        if self.log_classifications:
            logger.info(
                f" IntentConfig loaded: model={self.gemini_model}, "
                f"threshold={self.confidence_threshold}, timeout={self.gemini_timeout}s, "
                f"target_fast_route={self.fast_route_target}"
            )
    
    @staticmethod
    def from_env() -> "IntentConfig":
        """
        Load configuration from environment variables.
        
        Environment Variables:
            GEMINI_API_KEY: Required Gemini API key
            GEMINI_MODEL: Gemini model name (default: gemini-2.0-flash-lite)
            LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD: Fast route threshold (default: 0.8)
            LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT: Gemini timeout in seconds (default: 5.0)
            LEIBNIZ_INTENT_PARSER_ENABLE_CONTEXT_EXTRACTION: Enable context extraction (default: true)
            LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS: Log classifications (default: true)
            LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT: Strip non-essential fields (default: false)
            LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET: Target fast route percentage (default: 0.8)
        
        Returns:
            IntentConfig instance loaded from environment
        """
        # Load Gemini API key (required for LLM fallback)
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        # Load Gemini model name
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
        
        # Load confidence threshold with validation
        try:
            confidence_threshold = float(
                os.getenv("LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD", "0.8")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_PARSER_CONFIDENCE_THRESHOLD, using default 0.8"
            )
            confidence_threshold = 0.8
        
        # Load Gemini timeout with validation
        try:
            gemini_timeout = float(
                os.getenv("LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT", "5.0")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_PARSER_GEMINI_TIMEOUT, using default 5.0"
            )
            gemini_timeout = 5.0
        
        # Load boolean flags
        enable_context_extraction = os.getenv(
            "LEIBNIZ_INTENT_PARSER_ENABLE_CONTEXT_EXTRACTION", "true"
        ).lower() in ("true", "1", "yes")
        
        log_classifications = os.getenv(
            "LEIBNIZ_INTENT_PARSER_LOG_CLASSIFICATIONS", "true"
        ).lower() in ("true", "1", "yes")
        
        minimal_output = os.getenv(
            "LEIBNIZ_INTENT_PARSER_MINIMAL_OUTPUT", "false"
        ).lower() in ("true", "1", "yes")
        
        # Load fast route target with validation
        try:
            fast_route_target = float(
                os.getenv("LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET", "0.8")
            )
        except ValueError:
            logger.warning(
                "Invalid LEIBNIZ_INTENT_PARSER_FAST_ROUTE_TARGET, using default 0.8"
            )
            fast_route_target = 0.8
        
        return IntentConfig(
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            confidence_threshold=confidence_threshold,
            gemini_timeout=gemini_timeout,
            enable_context_extraction=enable_context_extraction,
            log_classifications=log_classifications,
            minimal_output=minimal_output,
            fast_route_target=fast_route_target,
        )
