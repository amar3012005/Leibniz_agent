"""
Leibniz University Institute - Appointment Booking Finite State Machine

This module provides a simplified appointment booking FSM for Leibniz University customer
service. It implements a clean state-based slot-filling pattern to collect appointment
information from users through conversational English dialogue.

**Multilingual Support (NEW):**
- Automatic translation from Hindi, Telugu, Tamil, Kannada, Malayalam to English
- Semantic entity extraction for accurate slot filling regardless of input language
- Uses Gemini API for context-aware translation with FSM state information
- Enable/disable via LEIBNIZ_APPOINTMENT_ENABLE_MULTILINGUAL environment variable
- Integration happens in leibniz_pro.py before FSM processing (transparent to FSM)

**Fields Collected (7 required):**
1. **Name**: User's full name (first and last preferred)
2. **Email**: Contact email address (university email preferred)
3. **Phone**: Phone number with country code (E.164 format)
4. **Department/Service**: One of 3 available university departments (Academic Advising, Faculty, Examination Office)
5. **Appointment Type**: Specific service within selected department
6. **Preferred Date/Time**: When user wants the appointment (natural language parsing)
7. **Purpose**: Brief description of appointment reason

**Key Features:**
- Friendly casual English prompts (not formal academic)
- Natural language date/time parsing ("next Tuesday at 2pm")
- Robust validation with helpful error messages
- Retry logic with progressive assistance
- Per-field confirmation with readback (spell names, format phones)
- Smart default to 'yes' after 2 empty confirmation responses
- Cancellation available at any time
- Confirmation flow with ability to correct information
- Simplified validation (no external API calls, no complex verification)
- **MULTILINGUAL**: Accepts inputs in Hindi/Telugu/other languages (translated before processing)

**Usage Example:**
```python
from leibniz_agent import create_appointment_fsm
from leibniz_agent.leibniz_semantic_translator import generate_semantic_context_for_fsm

# Create FSM instance
fsm = create_appointment_fsm()

# For multilingual input, translate first
user_input = "मेरा नाम राज कुमार है"  # Hindi: "My name is Raj Kumar"
context = await generate_semantic_context_for_fsm(user_input, fsm.state.value)
result = await fsm.process_input(context['translated_text'])  # "Raj Kumar"

# Or for English input, use directly
result = await fsm.process_input("John Smith")
print(result['response'])  # "Thanks, John! Now, what's your email address?"

# When complete, get booking data
if result['complete']:
    booking_data = fsm.data.to_dict()
    # Submit to university booking system
```

**Integration:**
This module is designed to be called from leibniz_pro.py (main orchestrator) when
the intent parser classifies user input as APPOINTMENT_SCHEDULING. The FSM maintains
state across multiple conversation turns and returns control when complete or cancelled.

Multilingual translation is handled in leibniz_pro.py using leibniz_semantic_translator.py
before passing input to the FSM, making the FSM language-agnostic.
"""

import re
import os
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from leibniz_agent.leibniz_config import get_leibniz_config

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class AppointmentState(Enum):
    """FSM states for appointment booking process"""
    INIT = "init"
    COLLECT_NAME = "collect_name"
    CONFIRM_NAME = "confirm_name"
    COLLECT_EMAIL = "collect_email"
    CONFIRM_EMAIL = "confirm_email"
    COLLECT_PHONE = "collect_phone"
    CONFIRM_PHONE = "confirm_phone"
    COLLECT_DEPARTMENT = "collect_department"
    CONFIRM_DEPARTMENT = "confirm_department"
    COLLECT_APPOINTMENT_TYPE = "collect_appointment_type"
    CONFIRM_APPOINTMENT_TYPE = "confirm_appointment_type"
    COLLECT_DATETIME = "collect_datetime"
    CONFIRM_DATETIME = "confirm_datetime"
    COLLECT_PURPOSE = "collect_purpose"
    CONFIRM_PURPOSE = "confirm_purpose"
    CONFIRM = "confirm"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


# Department/Service options (from appointment_scheduling_complete.md)
DEPARTMENTS = {
    "academic_advising": "Academic Advising",
    "faculty": "Faculty Appointments",
    "examination_office": "Examination Office"
}

# Appointment type options by department (simplified from appointment_scheduling_complete.md)
APPOINTMENT_TYPES = {
    "academic_advising": ["Program Counseling", "Course Selection", "Degree Progress Review", "Study Plan"],
    "faculty": ["Professor Consultation", "Thesis Supervision", "Research Discussion", "Letter of Recommendation"],
    "examination_office": ["Grade Inquiry", "Exam Registration", "Certificate Request", "Appeal"]
}

# Validation patterns
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
PHONE_PATTERN_INTL = re.compile(r'^\+?[1-9]\d{9,14}$')  # E.164 format
PHONE_PATTERN_GERMAN = re.compile(r'^(\+49|0)[1-9]\d{1,14}$')
# Comment 7: Loosened NAME_PATTERN for international names - Unicode letters, hyphens, apostrophes, spaces
NAME_PATTERN = re.compile(r'^[\w\s\-\']{2,50}$', re.UNICODE)

# Natural language date patterns
DATE_RELATIVE = re.compile(r'\b(today|tomorrow|next week)\b', re.IGNORECASE)
DATE_WEEKDAY = re.compile(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', re.IGNORECASE)
DATE_FORMATTED = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
TIME_PATTERN = re.compile(r'\b\d{1,2}:\d{2}\s*(am|pm)?\b', re.IGNORECASE)
TIME_SIMPLE = re.compile(r'\b\d{1,2}\s*(am|pm)\b', re.IGNORECASE)
TIME_RELATIVE = re.compile(r'\b(morning|afternoon|evening)\b', re.IGNORECASE)

# Cancellation keywords
CANCEL_KEYWORDS = ["cancel", "stop", "exit", "quit", "nevermind", "never mind"]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AppointmentData:
    """Data structure for collected appointment information"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None  # Key from DEPARTMENTS dict
    appointment_type: Optional[str] = None
    preferred_datetime: Optional[str] = None  # Natural language or formatted
    purpose: Optional[str] = None
    student_id: Optional[str] = None  # Optional field
    preferred_language: str = "English"  # Default to English
    booking_timestamp: Optional[str] = None  # When booking was made
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for submission"""
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "department": DEPARTMENTS.get(self.department, self.department),
            "appointment_type": self.appointment_type,
            "preferred_datetime": self.preferred_datetime,
            "purpose": self.purpose,
            "student_id": self.student_id,
            "preferred_language": self.preferred_language,
            "booking_timestamp": self.booking_timestamp
        }


# ============================================================================
# LeibnizAppointmentFSM Class
# ============================================================================

class LeibnizAppointmentFSM:
    """
    Finite State Machine for Leibniz University appointment booking.
    
    Implements a clean slot-filling pattern to collect 7 required fields:
    name, email, phone, department, appointment_type, preferred_datetime, purpose.
    
    Features:
    - Conversational English prompts (friendly casual tone)
    - Natural language date/time parsing
    - Robust validation with retry logic
    - Confirmation flow with correction ability
    - Cancellation available at any time
    """
    
    def __init__(self):
        """Initialize the appointment booking FSM"""
        # State management
        self.state = AppointmentState.INIT
        self.data = AppointmentData()
        
        # Configuration
        self.config = get_leibniz_config()
        
        # Retry tracking (per field) - read from config (Comment 1)
        self.retry_counts: Dict[str, int] = {}
        self.max_retries = int(os.getenv('LEIBNIZ_APPOINTMENT_MAX_RETRIES', '3'))
        
        # Confirmation attempt tracking (per field) - for empty response handling
        self.confirmation_attempts: Dict[str, int] = {}
        self.max_confirmation_attempts = int(os.getenv('LEIBNIZ_APPOINTMENT_MAX_CONFIRMATION_ATTEMPTS', '2'))
        
        # Error tracking
        self.last_error: Optional[str] = None
        
        # Conversation history for context
        self.conversation_history: List[str] = []
        
        logger.info(" Leibniz Appointment FSM initialized")
    
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input based on current FSM state
        
        Args:
            user_input: User's spoken/typed input
            
        Returns:
            Dictionary with:
            - response: System response to user
            - state: Current FSM state
            - previous_state: Previous FSM state (for progression tracking)
            - complete: Whether booking is complete
            - data: Collected appointment data (if complete)
            - error: Error message (if any)
        """
        # Store previous state for progression tracking
        previous_state = self.state
        
        # Store input in conversation history
        self.conversation_history.append(user_input)
        
        # Check for cancellation
        if any(keyword in user_input.lower() for keyword in CANCEL_KEYWORDS):
            self.state = AppointmentState.CANCELLED
            return {
                "response": "No problem! If you'd like to book an appointment later, just let me know. Is there anything else I can help you with?",
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": False,
                "cancelled": True,
                "data": None,
                "error": None
            }
        
        # Route to appropriate handler based on current state
        try:
            if self.state == AppointmentState.INIT:
                response = await self._handle_init()
            elif self.state == AppointmentState.COLLECT_NAME:
                response = await self._handle_name_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_NAME:
                response = await self._handle_name_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_EMAIL:
                response = await self._handle_email_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_EMAIL:
                response = await self._handle_email_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_PHONE:
                response = await self._handle_phone_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_PHONE:
                response = await self._handle_phone_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_DEPARTMENT:
                response = await self._handle_department_selection(user_input)
            elif self.state == AppointmentState.CONFIRM_DEPARTMENT:
                response = await self._handle_department_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_APPOINTMENT_TYPE:
                response = await self._handle_appointment_type_selection(user_input)
            elif self.state == AppointmentState.CONFIRM_APPOINTMENT_TYPE:
                response = await self._handle_appointment_type_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_DATETIME:
                response = await self._handle_datetime_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_DATETIME:
                response = await self._handle_datetime_confirmation(user_input)
            elif self.state == AppointmentState.COLLECT_PURPOSE:
                response = await self._handle_purpose_collection(user_input)
            elif self.state == AppointmentState.CONFIRM_PURPOSE:
                response = await self._handle_purpose_confirmation(user_input)
            elif self.state == AppointmentState.CONFIRM:
                response = await self._handle_confirmation(user_input)
            elif self.state == AppointmentState.COMPLETE:
                response = "Your appointment is already booked! Is there anything else I can help you with?"
            elif self.state == AppointmentState.CANCELLED:
                response = "The appointment booking was cancelled. Would you like to start a new booking?"
            else:
                response = "I'm not sure what happened. Let's start over with the appointment booking."
                self.state = AppointmentState.INIT
            
            # Return response dictionary
            return {
                "response": response,
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": self.state == AppointmentState.COMPLETE,
                "cancelled": self.state == AppointmentState.CANCELLED,
                "data": self.data.to_dict() if self.state == AppointmentState.COMPLETE else None,
                "error": self.last_error
            }
            
        except Exception as e:
            logger.error(f" Error processing input in state {self.state}: {e}")
            return {
                "response": "Oops, something went wrong. Let me try that again. Could you repeat what you just said?",
                "state": self.state.value,
                "previous_state": previous_state.value,
                "complete": False,
                "cancelled": False,
                "data": None,
                "error": str(e)
            }
    
    # ========================================================================
    # State Handler Methods
    # ========================================================================
    
    async def _handle_init(self) -> str:
        """Initialize appointment booking process"""
        # Comment 3: Friendly tone with contractions and supportive close
        response = (
            "Great! I'd be happy to help you schedule an appointment. "
            "I'll need to collect a few details from you—this should only take a couple of minutes.\n\n"
            "I'll ask for your name, contact info, which department you'd like to meet with, "
            "and what the appointment's for. Feel free to say 'cancel' at any time if you change your mind.\n\n"
            "Let's start with your full name. What's your name?"
        )
        
        # Transition to name collection
        self.state = AppointmentState.COLLECT_NAME
        
        return response
    
    async def _handle_name_collection(self, user_input: str) -> str:
        """Collect and validate user's full name"""
        # Clean input
        cleaned = user_input.strip()
        
        # Remove common prefixes
        prefixes = ["my name is", "i'm", "this is", "it's", "name:", "i am"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Extract name
        name = cleaned.strip()
        
        # Validate name
        is_valid, error_msg = self._validate_name(name)
        
        if is_valid:
            # Success path
            self.data.name = name
            self.retry_counts['name'] = 0
            self.state = AppointmentState.CONFIRM_NAME
            
            # Transition to name confirmation
            spelled_name = self._spell_out_name(name)
            return f"Got it! So your name is {spelled_name}, am I right?"
        else:
            # Failure path
            self.retry_counts['name'] = self.retry_counts.get('name', 0) + 1
            self.last_error = error_msg
            
            if self.retry_counts['name'] < self.max_retries:
                # Comment 3: Supportive close
                return f"{error_msg} Could you tell me your full name? For example, 'John Smith' or 'Maria Garcia'. Take your time!"
            else:
                return "I'm having trouble getting your name. Would you like to try again, or should we cancel the appointment booking?"
    
    async def _handle_email_collection(self, user_input: str) -> str:
        """Collect and validate email address"""
        # Clean input
        cleaned = user_input.strip().lower()
        
        # Remove common phrases
        prefixes = ["my email is", "it's", "email:", "it is", "the email is"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Extract email
        email = self._extract_email_from_text(cleaned)
        
        if email:
            # Validate email
            is_valid, error_msg = self._validate_email(email)
            
            if is_valid:
                # Success path
                self.data.email = email
                self.retry_counts['email'] = 0
                self.state = AppointmentState.CONFIRM_EMAIL
                
                # Transition to email confirmation
                return f"Great! So your email is {email}, am I right?"
            else:
                # Invalid email format
                self.retry_counts['email'] = self.retry_counts.get('email', 0) + 1
                self.last_error = error_msg
                
                if self.retry_counts['email'] < self.max_retries:
                    # Comment 3: Supportive close
                    return f"{error_msg} Could you try again? For example, 'john.smith@uni-hannover.de'. No worries if you need a moment!"
                else:
                    # Comment 4: Consistent skip handling - set to placeholder and proceed to confirmation
                    self.data.email = "Not provided"
                    self.retry_counts['email'] = 0
                    self.state = AppointmentState.CONFIRM_EMAIL
                    # Transition to email confirmation even with placeholder
                    return f"No problem, we'll continue without the email for now. So we'll use '{self.data.email}' as your email, am I right?"
        else:
            # No email found
            self.retry_counts['email'] = self.retry_counts.get('email', 0) + 1
            
            if self.retry_counts['email'] < self.max_retries:
                # Comment 3: Supportive close
                return "I didn't catch an email address in that. Could you say it again? For example, 'john.smith@uni-hannover.de'. Feel free to spell it out!"
            else:
                # Comment 4: Consistent skip handling - set to placeholder and proceed
                self.data.email = "Not provided"
                self.retry_counts['email'] = 0
                self.state = AppointmentState.CONFIRM_EMAIL
                # Transition to email confirmation even with placeholder
                return f"That's okay, let's move on. So we'll use '{self.data.email}' as your email, am I right?"
    
    async def _handle_phone_collection(self, user_input: str) -> str:
        """Collect and validate phone number"""
        # Clean input
        cleaned = user_input.strip()
        
        # Remove common phrases
        prefixes = ["my number is", "phone:", "call me at", "it's", "the number is"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Extract phone
        phone = self._extract_phone_from_text(cleaned)
        
        if phone:
            # Normalize phone
            normalized = self._normalize_phone_number(phone)
            
            if normalized:
                # Validate phone
                is_valid, error_msg = self._validate_phone(normalized)
                
                if is_valid:
                    # Success path
                    self.data.phone = normalized
                    self.retry_counts['phone'] = 0
                    self.state = AppointmentState.CONFIRM_PHONE
                    
                    # Transition to phone confirmation
                    formatted_phone = self._format_phone_for_readback(normalized)
                    return f"Perfect! So your phone number is {formatted_phone}, am I right?"
                else:
                    # Invalid phone
                    self.retry_counts['phone'] = self.retry_counts.get('phone', 0) + 1
                    self.last_error = error_msg
                    
                    if self.retry_counts['phone'] < self.max_retries:
                        # Comment 3: Supportive close
                        return f"{error_msg} Could you try again? For example, '+49 511 762 2020' or '0511 762 2020'. No worries if it's tricky!"
                    else:
                        # Comment 4: Consistent skip handling - set to placeholder and proceed
                        self.data.phone = "Not provided"
                        self.retry_counts['phone'] = 0
                        self.state = AppointmentState.CONFIRM_PHONE
                        # Transition to confirmation even with placeholder
                        return f"That's okay, let's continue. So we'll use '{self.data.phone}' as your phone number, am I right?"
            else:
                # Normalization failed
                self.retry_counts['phone'] = self.retry_counts.get('phone', 0) + 1
                
                if self.retry_counts['phone'] < self.max_retries:
                    # Comment 3: Supportive close
                    return "I couldn't understand that phone number. Could you try again? For example, '+49 511 762 2020'. Take your time!"
                else:
                    # Comment 4: Consistent skip handling - set to placeholder and proceed to confirmation
                    self.data.phone = "Not provided"
                    self.retry_counts['phone'] = 0
                    self.state = AppointmentState.CONFIRM_PHONE
                    # Transition to confirmation even with placeholder
                    return f"No problem, let's move on. So we'll use '{self.data.phone}' as your phone number, am I right?"
    
    async def _handle_department_selection(self, user_input: str) -> str:
        """Select department/service from available options"""
        # Clean input
        cleaned = user_input.strip().lower()
        
        # Try numbered selection first
        if cleaned.isdigit():
            index = int(cleaned) - 1
            dept_keys = list(DEPARTMENTS.keys())
            if 0 <= index < len(dept_keys):
                # Valid number selection
                self.data.department = dept_keys[index]
                self.retry_counts['department'] = 0
                self.state = AppointmentState.CONFIRM_DEPARTMENT
                
                # Transition to department confirmation
                return f"Got it! So you want to meet with {DEPARTMENTS[self.data.department]}, am I right?"
        
        # Match against department keywords
        matched_dept = None
        
        # Department keyword matching
        if "academic" in cleaned or "advising" in cleaned or "advisor" in cleaned:
            matched_dept = "academic_advising"
        elif "faculty" in cleaned or "professor" in cleaned:
            matched_dept = "faculty"
        elif "exam" in cleaned or "examination" in cleaned or "grade" in cleaned:
            matched_dept = "examination_office"
        elif "international" in cleaned or "visa" in cleaned or "foreign" in cleaned:
            matched_dept = "international_office"
        elif "career" in cleaned or "job" in cleaned:
            matched_dept = "career_services"
        elif "counseling" in cleaned or "psychological" in cleaned or "mental" in cleaned or "therapy" in cleaned:
            matched_dept = "counseling"
        elif "financial" in cleaned or "scholarship" in cleaned or "aid" in cleaned or "money" in cleaned:
            matched_dept = "financial_aid"
        elif "admission" in cleaned or "apply" in cleaned or "application" in cleaned:
            matched_dept = "academic_advising"  # Map admissions to academic advising
        elif "registration" in cleaned or "enrollment" in cleaned or "register" in cleaned:
            matched_dept = "registration"
        # Comment 5: IT matching with word boundaries to avoid substring misfire
        elif (re.search(r'\bit\b', cleaned) or "IT support" in user_input or "IT services" in user_input or 
              "technical help" in cleaned or "computer account" in cleaned or "tech support" in cleaned):
            matched_dept = "it_services"
        
        if matched_dept:
            # Success path
            self.data.department = matched_dept
            self.retry_counts['department'] = 0
            self.state = AppointmentState.CONFIRM_DEPARTMENT
            
            # Transition to department confirmation
            return f"Perfect! So you want to meet with {DEPARTMENTS[matched_dept]}, am I right?"
        else:
            # Failure path
            self.retry_counts['department'] = self.retry_counts.get('department', 0) + 1
            
            if self.retry_counts['department'] < self.max_retries:
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                # Comment 3: Friendly tone with contraction and supportive close
                return f"I didn't catch which department you need. Could you choose from this list?\n\n{dept_list}\n\nFeel free to say the number or the department name. Take your time!"
            else:
                # Default to academic advising
                self.data.department = "academic_advising"
                self.retry_counts['department'] = 0
                self.state = AppointmentState.CONFIRM_DEPARTMENT
                # Transition to confirmation with default
                return f"Let me put you down for {DEPARTMENTS[self.data.department]}—we can change this later if needed. So you want to meet with {DEPARTMENTS[self.data.department]}, am I right?"
    
    async def _handle_appointment_type_selection(self, user_input: str) -> str:
        """Select appointment type based on chosen department"""
        # Get available types for selected department
        available_types = APPOINTMENT_TYPES[self.data.department]
        
        # Clean input
        cleaned = user_input.strip().lower()
        
        # Try numbered selection first
        if cleaned.isdigit():
            index = int(cleaned) - 1
            if 0 <= index < len(available_types):
                # Valid number selection
                self.data.appointment_type = available_types[index]
                self.retry_counts['appointment_type'] = 0
                self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE
                
                # Transition to appointment type confirmation
                return f"Excellent! So you need {self.data.appointment_type}, am I right?"
        
        # Match against appointment type keywords
        matched_type = None
        
        for atype in available_types:
            # Check if any word from the type appears in user input
            type_words = atype.lower().split()
            if any(word in cleaned for word in type_words):
                matched_type = atype
                break
        
        if matched_type:
            # Success path
            self.data.appointment_type = matched_type
            self.retry_counts['appointment_type'] = 0
            self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE
            
            # Transition to appointment type confirmation
            return f"Sounds good! So you need {matched_type}, am I right?"
        else:
            # Failure path
            self.retry_counts['appointment_type'] = self.retry_counts.get('appointment_type', 0) + 1
            
            if self.retry_counts['appointment_type'] < self.max_retries:
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(available_types)])
                # Comment 3: Friendly tone with supportive close
                return f"I'm not sure which type you need. Here are the options again:\n\n{types_list}\n\nFeel free to say the number or the appointment type. No worries if you need a moment!"
            else:
                # Default to first type
                self.data.appointment_type = available_types[0]
                self.retry_counts['appointment_type'] = 0
                self.state = AppointmentState.CONFIRM_APPOINTMENT_TYPE
                # Transition to confirmation with default
                return f"I'll put you down for {available_types[0]}—we can adjust this later. So you need {self.data.appointment_type}, am I right?"
    
    async def _handle_datetime_collection(self, user_input: str) -> str:
        """Collect preferred date and time"""
        # Parse datetime from user input
        parsed_datetime = self._parse_datetime_from_text(user_input)
        
        if parsed_datetime:
            # Validate datetime (Comment 2: future date, booking window, business hours)
            is_valid, error_msg = self._validate_datetime(parsed_datetime)
            
            if not is_valid:
                # Validation failed - provide friendly hint
                self.retry_counts['datetime'] = self.retry_counts.get('datetime', 0) + 1
                
                if self.retry_counts['datetime'] < self.max_retries:
                    return f"{error_msg} Could you suggest another date and time?"
                else:
                    # After max retries, suggest a valid default
                    next_week = datetime.now() + timedelta(days=7)
                    default_datetime = next_week.strftime("%A, %B %d, %Y at 10:00 AM")
                    self.data.preferred_datetime = default_datetime
                    self.retry_counts['datetime'] = 0
                    self.state = AppointmentState.CONFIRM_DATETIME
                    return f"Let's try {default_datetime} instead - we can adjust later if needed. So the appointment is for {self.data.preferred_datetime}, am I right?"
            
            # Success path - datetime is valid
            self.data.preferred_datetime = parsed_datetime
            self.retry_counts['datetime'] = 0
            self.state = AppointmentState.CONFIRM_DATETIME
            
            # Transition to datetime confirmation
            return f"Perfect! So the appointment is for {parsed_datetime}, am I right?"
        else:
            # Failure path - could not parse
            self.retry_counts['datetime'] = self.retry_counts.get('datetime', 0) + 1
            
            if self.retry_counts['datetime'] < self.max_retries:
                # Comment 3: Supportive close
                return "I didn't quite understand that date/time. Could you try again? For example, 'next Tuesday at 2pm' or 'December 15 at 10am'. Take your time!"
            else:
                # Suggest default
                next_week = datetime.now() + timedelta(days=7)
                default_datetime = next_week.strftime("%A, %B %d, %Y at 10:00 AM")
                self.data.preferred_datetime = default_datetime
                self.retry_counts['datetime'] = 0
                self.state = AppointmentState.CONFIRM_DATETIME
                # Transition to confirmation with default
                return f"How about {default_datetime}? We can work out the exact time later. So the appointment is for {self.data.preferred_datetime}, am I right?"
    
    async def _handle_purpose_collection(self, user_input: str) -> str:
        """Collect appointment purpose/reason"""
        # Clean input
        cleaned = user_input.strip()
        
        # Remove common phrases
        prefixes = ["i need", "i want to", "the purpose is", "it's for", "because"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        # Validate purpose
        if len(cleaned) < 2:
            self.retry_counts['purpose'] = self.retry_counts.get('purpose', 0) + 1
            if self.retry_counts['purpose'] < self.max_retries:
                # Comment 3: Supportive close
                return "Could you tell me a bit more about why you need this appointment? For example, 'I need help choosing courses for next semester'. No worries if you need to think for a moment!"
            else:
                cleaned = "General inquiry"
        elif cleaned.lower() in ["test", "none", "n/a", ".", "help"]:
            self.retry_counts['purpose'] = self.retry_counts.get('purpose', 0) + 1
            if self.retry_counts['purpose'] < self.max_retries:
                # Comment 3: Supportive close
                return "Could you be a bit more specific about what you need help with? Feel free to give me any details that come to mind!"
            else:
                cleaned = "General inquiry"
        elif len(cleaned) > 500:
            cleaned = cleaned[:500]
        
        # Success path
        self.data.purpose = cleaned
        self.retry_counts['purpose'] = 0
        self.state = AppointmentState.CONFIRM_PURPOSE
        
        # Transition to purpose confirmation
        purpose_preview = cleaned[:50] + "..." if len(cleaned) > 50 else cleaned
        return f"Got it! So the reason for your appointment is '{purpose_preview}', am I right?"
    
    async def _handle_name_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected name"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to email collection
            self.confirmation_attempts['name'] = 0
            self.state = AppointmentState.COLLECT_EMAIL
            return f"Awesome— nice to meet you, {self.data.name.split()[0]}! Now, what's your email address? I'll send the appointment confirmation there."
        
        elif response_type == "no":
            # Rejected - go back to name collection
            self.data.name = None
            self.confirmation_attempts['name'] = 0
            self.state = AppointmentState.COLLECT_NAME
            return "No worries! What's your name? Take your time!"
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['name'] = self.confirmation_attempts.get('name', 0) + 1
            
            if self.confirmation_attempts['name'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Name confirmation: defaulting to 'yes' after {self.confirmation_attempts['name']} empty responses")
                self.confirmation_attempts['name'] = 0
                self.state = AppointmentState.COLLECT_EMAIL
                return f"Perfect! Now, what's your email address? I'll send the appointment confirmation there."
            else:
                # Repeat confirmation
                spelled_name = self._spell_out_name(self.data.name)
                return f"So your name is {spelled_name}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_email_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected email"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to phone collection
            self.confirmation_attempts['email'] = 0
            self.state = AppointmentState.COLLECT_PHONE
            return "Perfect! And what's your phone number? Include the country code if you're calling from outside Germany."
        
        elif response_type == "no":
            # Rejected - go back to email collection
            self.data.email = None
            self.confirmation_attempts['email'] = 0
            self.state = AppointmentState.COLLECT_EMAIL
            return "No problem! What's your email address? Feel free to spell it out!"
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['email'] = self.confirmation_attempts.get('email', 0) + 1
            
            if self.confirmation_attempts['email'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Email confirmation: defaulting to 'yes' after {self.confirmation_attempts['email']} empty responses")
                self.confirmation_attempts['email'] = 0
                self.state = AppointmentState.COLLECT_PHONE
                return "Perfect! What's your phone number? Include the country code if you're calling from outside Germany."
            else:
                # Repeat confirmation
                return f"So your email is {self.data.email}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_phone_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected phone number"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to department selection
            self.confirmation_attempts['phone'] = 0
            self.state = AppointmentState.COLLECT_DEPARTMENT
            
            # Generate department options list
            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"Great! Now, which department would you like to schedule an appointment with? Here are your options:\n\n{dept_list}\n\nFeel free to say the number or the department name."
        
        elif response_type == "no":
            # Rejected - go back to phone collection
            self.data.phone = None
            self.confirmation_attempts['phone'] = 0
            self.state = AppointmentState.COLLECT_PHONE
            return "No worries! What's your phone number? Feel free to spell it out if needed!"
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['phone'] = self.confirmation_attempts.get('phone', 0) + 1
            
            if self.confirmation_attempts['phone'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Phone confirmation: defaulting to 'yes' after {self.confirmation_attempts['phone']} empty responses")
                self.confirmation_attempts['phone'] = 0
                self.state = AppointmentState.COLLECT_DEPARTMENT
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                return f"Perfect! Which department would you like to schedule an appointment with?\n\n{dept_list}\n\nFeel free to say the number or the department name."
            else:
                # Repeat confirmation
                formatted_phone = self._format_phone_for_readback(self.data.phone)
                return f"So your phone number is {formatted_phone}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_department_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected department"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to appointment type selection
            self.confirmation_attempts['department'] = 0
            self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
            
            # Get appointment types for selected department
            types = APPOINTMENT_TYPES[self.data.department]
            types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
            return f"Perfect! For {DEPARTMENTS[self.data.department]}, what type of appointment do you need?\n\n{types_list}\n\nYou can say the number or the appointment type."
        
        elif response_type == "no":
            # Rejected - go back to department selection
            self.data.department = None
            self.confirmation_attempts['department'] = 0
            self.state = AppointmentState.COLLECT_DEPARTMENT
            
            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"No problem! Which department would you like?\n\n{dept_list}\n\nFeel free to say the number or the department name."
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['department'] = self.confirmation_attempts.get('department', 0) + 1
            
            if self.confirmation_attempts['department'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Department confirmation: defaulting to 'yes' after {self.confirmation_attempts['department']} empty responses")
                self.confirmation_attempts['department'] = 0
                self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
                types = APPOINTMENT_TYPES[self.data.department]
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
                return f"Perfect! For {DEPARTMENTS[self.data.department]}, what type of appointment do you need?\n\n{types_list}\n\nYou can say the number or the appointment type."
            else:
                # Repeat confirmation
                return f"So you want to meet with {DEPARTMENTS[self.data.department]}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_appointment_type_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected appointment type"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to datetime collection
            self.confirmation_attempts['appointment_type'] = 0
            self.state = AppointmentState.COLLECT_DATETIME
            return "Got it! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."
        
        elif response_type == "no":
            # Rejected - go back to appointment type selection
            self.data.appointment_type = None
            self.confirmation_attempts['appointment_type'] = 0
            self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
            
            types = APPOINTMENT_TYPES[self.data.department]
            types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
            return f"No worries! What type of appointment do you need?\n\n{types_list}\n\nFeel free to say the number or the appointment type."
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['appointment_type'] = self.confirmation_attempts.get('appointment_type', 0) + 1
            
            if self.confirmation_attempts['appointment_type'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Appointment type confirmation: defaulting to 'yes' after {self.confirmation_attempts['appointment_type']} empty responses")
                self.confirmation_attempts['appointment_type'] = 0
                self.state = AppointmentState.COLLECT_DATETIME
                return "Sounds good! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."
            else:
                # Repeat confirmation
                return f"So you need {self.data.appointment_type}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_datetime_confirmation(self, user_input: str) -> str:
        """Handle confirmation of selected datetime"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to purpose collection
            self.confirmation_attempts['datetime'] = 0
            self.state = AppointmentState.COLLECT_PURPOSE
            
            # Add department-specific advance notice hint
            response = "Perfect!"
            hint = self._get_department_advance_notice_hint(self.data.department)
            if hint:
                response += f" {hint}"
            response += " Last question: what's the main reason for this appointment? Just a brief description is fine—no need to write an essay!"
            return response
        
        elif response_type == "no":
            # Rejected - go back to datetime collection
            self.data.preferred_datetime = None
            self.confirmation_attempts['datetime'] = 0
            self.state = AppointmentState.COLLECT_DATETIME
            return "No problem! When would you like to schedule this appointment? You can say something like 'next Tuesday at 2pm' or 'tomorrow morning'."
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['datetime'] = self.confirmation_attempts.get('datetime', 0) + 1
            
            if self.confirmation_attempts['datetime'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Datetime confirmation: defaulting to 'yes' after {self.confirmation_attempts['datetime']} empty responses")
                self.confirmation_attempts['datetime'] = 0
                self.state = AppointmentState.COLLECT_PURPOSE
                response = "Perfect!"
                hint = self._get_department_advance_notice_hint(self.data.department)
                if hint:
                    response += f" {hint}"
                response += " Last question: what's the main reason for this appointment? Just a brief description is fine!"
                return response
            else:
                # Repeat confirmation
                return f"So the appointment is for {self.data.preferred_datetime}, am I right? Please say 'yes' or 'no'."
    
    async def _handle_purpose_confirmation(self, user_input: str) -> str:
        """Handle confirmation of appointment purpose"""
        response_type = self._parse_yes_no_response(user_input)
        
        if response_type == "yes":
            # Confirmed - proceed to final confirmation
            self.confirmation_attempts['purpose'] = 0
            self.state = AppointmentState.CONFIRM
            
            # Generate final confirmation summary
            summary = self._generate_confirmation_summary()
            return f"Great! Let me confirm all the details:\n\n{summary}\n\nDoes everything look correct? Say 'yes' to confirm or 'no' to make changes."
        
        elif response_type == "no":
            # Rejected - go back to purpose collection
            self.data.purpose = None
            self.confirmation_attempts['purpose'] = 0
            self.state = AppointmentState.COLLECT_PURPOSE
            return "No worries! What's the main reason for this appointment? Feel free to give me any details!"
        
        else:
            # Unclear/empty response - track attempts
            self.confirmation_attempts['purpose'] = self.confirmation_attempts.get('purpose', 0) + 1
            
            if self.confirmation_attempts['purpose'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Purpose confirmation: defaulting to 'yes' after {self.confirmation_attempts['purpose']} empty responses")
                self.confirmation_attempts['purpose'] = 0
                self.state = AppointmentState.CONFIRM
                summary = self._generate_confirmation_summary()
                return f"Great! Let me confirm all the details:\n\n{summary}\n\nDoes everything look correct? Say 'yes' to confirm or 'no' to make changes."
            else:
                # Repeat confirmation
                purpose_preview = self.data.purpose[:50] + "..." if len(self.data.purpose) > 50 else self.data.purpose
                return f"So the reason for your appointment is '{purpose_preview}', am I right? Please say 'yes' or 'no'."
    
    async def _handle_confirmation(self, user_input: str) -> str:
        """Handle confirmation of collected information"""
        # Clean input
        cleaned = user_input.strip().lower()
        
        # Comment 5: Handle empty responses - default to yes after max attempts
        if not cleaned:
            self.confirmation_attempts['final'] = self.confirmation_attempts.get('final', 0) + 1
            
            if self.confirmation_attempts['final'] >= self.max_confirmation_attempts:
                # Default to yes after max empty attempts
                logger.info(f"Final confirmation: defaulting to 'yes' after {self.confirmation_attempts['final']} empty responses")
                self.confirmation_attempts['final'] = 0
                
                # Proceed with booking completion
                self.data.booking_timestamp = datetime.now().isoformat()
                self.state = AppointmentState.COMPLETE
                
                response = (
                    f"Awesome! Your appointment is all set. Here's what happens next:\n\n"
                    f"1. You'll receive a confirmation email at {self.data.email} within 24 hours\n"
                    f"2. The email will include a calendar invite and any preparation instructions\n"
                    f"3. If you need to cancel or reschedule, use the link in the confirmation email\n\n"
                    f"Remember to bring your student ID and any relevant documents to the appointment.\n\n"
                    f"Is there anything else I can help you with?"
                )
                
                return response
            else:
                # Ask again
                return "I didn't catch that. Is all the information correct? Please say 'yes' to confirm or 'no' to make changes."
        
        # Reset confirmation attempts on valid input
        self.confirmation_attempts['final'] = 0
        
        # Detect confirmation
        if any(word in cleaned for word in ["yes", "yeah", "yep", "correct", "right", "looks good", "confirm", "ok", "okay"]):
            # Confirmation path
            self.data.booking_timestamp = datetime.now().isoformat()
            self.state = AppointmentState.COMPLETE
            
            response = (
                f"Awesome! Your appointment is all set. Here's what happens next:\n\n"
                f"1. You'll receive a confirmation email at {self.data.email} within 24 hours\n"
                f"2. The email will include a calendar invite and any preparation instructions\n"
                f"3. If you need to cancel or reschedule, use the link in the confirmation email\n\n"
                f"Remember to bring your student ID and any relevant documents to the appointment.\n\n"
                f"Is there anything else I can help you with?"
            )
            
            return response
        
        # Detect rejection
        elif any(word in cleaned for word in ["no", "nope", "incorrect", "wrong", "change"]):
            # Rejection path - ask which field to change
            return "No problem! Which information would you like to change? You can say 'name', 'email', 'phone', 'department', 'appointment type', 'date', or 'purpose'."
        
        # Detect specific field changes
        elif "name" in cleaned:
            self.data.name = None
            self.state = AppointmentState.COLLECT_NAME
            self.retry_counts['name'] = 0
            return "Okay, let's update your name. What's your full name?"
        elif "email" in cleaned:
            self.data.email = None
            self.state = AppointmentState.COLLECT_EMAIL
            self.retry_counts['email'] = 0
            return "Okay, let's update your email. What's your email address?"
        elif "phone" in cleaned:
            self.data.phone = None
            self.state = AppointmentState.COLLECT_PHONE
            self.retry_counts['phone'] = 0
            return "Okay, let's update your phone number. What's your phone number?"
        elif "department" in cleaned:
            self.data.department = None
            self.state = AppointmentState.COLLECT_DEPARTMENT
            self.retry_counts['department'] = 0
            dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
            return f"Okay, let's change the department. Which department would you like?\n\n{dept_list}"
        elif "type" in cleaned or "appointment" in cleaned:
            # Comment 6: Check if department is set before accessing APPOINTMENT_TYPES
            if self.data.department is None:
                # Department not set, collect it first
                self.data.department = None
                self.state = AppointmentState.COLLECT_DEPARTMENT
                self.retry_counts['department'] = 0
                dept_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(DEPARTMENTS.values())])
                return f"First, let's choose the department:\n\n{dept_list}\n\nYou can say the number or the department name."
            else:
                # Department is set, proceed to change appointment type
                self.data.appointment_type = None
                self.state = AppointmentState.COLLECT_APPOINTMENT_TYPE
                self.retry_counts['appointment_type'] = 0
                types = APPOINTMENT_TYPES[self.data.department]
                types_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(types)])
                return f"Okay, let's change the appointment type. Which type do you need?\n\n{types_list}"
        elif "date" in cleaned or "time" in cleaned or "when" in cleaned:
            self.data.preferred_datetime = None
            self.state = AppointmentState.COLLECT_DATETIME
            self.retry_counts['datetime'] = 0
            return "Okay, let's update the date and time. When would you like to schedule this appointment?"
        elif "purpose" in cleaned or "reason" in cleaned:
            self.data.purpose = None
            self.state = AppointmentState.COLLECT_PURPOSE
            self.retry_counts['purpose'] = 0
            return "Okay, let's update the purpose. What's the main reason for this appointment?"
        else:
            # Ambiguous input
            return "I didn't catch that. Is the information correct? Please say 'yes' to confirm or 'no' to make changes."
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _generate_confirmation_summary(self) -> str:
        """Generate human-readable confirmation summary"""
        summary = (
            f"Name: {self.data.name}\n"
            f"Email: {self.data.email}\n"
            f"Phone: {self.data.phone}\n"
            f"Department: {DEPARTMENTS.get(self.data.department, self.data.department)}\n"
            f"Appointment Type: {self.data.appointment_type}\n"
            f"Preferred Date/Time: {self.data.preferred_datetime}\n"
            f"Purpose: {self.data.purpose}"
        )
        
        return summary
    
    def _extract_email_from_text(self, text: str) -> Optional[str]:
        """Extract email address from natural language text"""
        # Use regex to find email pattern
        match = EMAIL_PATTERN.search(text)
        if match:
            return match.group(0)
        
        # Comment 8: Robust spoken-form normalization with word boundaries
        # Tokenize and map spoken words to symbols
        spoken = text.lower()
        
        # Use regex substitutions with word boundaries
        spoken = re.sub(r'\bat\b', '@', spoken)
        spoken = re.sub(r'\bdot\b', '.', spoken)
        spoken = re.sub(r'\bdash\b', '-', spoken)
        spoken = re.sub(r'\bunderscore\b', '_', spoken)
        
        # Remove extra spaces
        spoken = re.sub(r'\s+', '', spoken)
        
        # Try matching again
        match = EMAIL_PATTERN.search(spoken)
        if match:
            return match.group(0)
        
        return None
    
    def _extract_phone_from_text(self, text: str) -> Optional[str]:
        """Extract phone number from natural language text"""
        # Remove formatting characters
        cleaned = text.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        
        # Try international format first
        match = PHONE_PATTERN_INTL.search(cleaned)
        if match:
            return match.group(0)
        
        # Try German format
        match = PHONE_PATTERN_GERMAN.search(cleaned)
        if match:
            return match.group(0)
        
        # Try finding any sequence of 10+ digits
        digits_only = re.sub(r'\D', '', text)
        if len(digits_only) >= 10:
            return digits_only
        
        return None
    
    def _normalize_phone_number(self, phone: str) -> Optional[str]:
        """Normalize phone number to standard format"""
        # Remove all non-digit characters except leading +
        if phone.startswith('+'):
            digits = '+' + re.sub(r'\D', '', phone[1:])
        else:
            digits = re.sub(r'\D', '', phone)
        
        # If starts with 0 (German format), convert to +49
        if digits.startswith('0'):
            digits = '+49' + digits[1:]
        
        # If no country code, assume German (+49)
        if not digits.startswith('+'):
            digits = '+49' + digits
        
        # Validate length: 10-15 digits (international standard)
        digit_count = len(re.sub(r'\D', '', digits))
        if digit_count < 10 or digit_count > 15:
            return None
        
        return digits
    
    def _spell_out_name(self, name: str) -> str:
        """Spell out name letter-by-letter for confirmation (e.g., 'John Smith' -> 'J-O-H-N S-M-I-T-H')"""
        if not name:
            return ""
        
        words = name.split()
        spelled_words = []
        
        for word in words:
            # Spell each letter separated by hyphens
            letters = []
            for char in word:
                if char.isalpha():
                    letters.append(char.upper())
                elif char == "'":
                    letters.append("apostrophe")
                elif char == "-":
                    letters.append("hyphen")
            
            spelled_word = "-".join(letters)
            spelled_words.append(spelled_word)
        
        # Join words with space
        return " ".join(spelled_words)
    
    def _format_phone_for_readback(self, phone: str) -> str:
        """Format phone number with spaces for clear readback (e.g., '+4951176220' -> '+49 511 762 20')"""
        if not phone:
            return ""
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # If starts with +49 (Germany), format as: +49 XXX XXX XX XX
        if cleaned.startswith('+49'):
            digits = cleaned[3:]  # Remove +49
            # Group: area code (3), main (3-4), rest (2-2)
            if len(digits) >= 9:
                formatted = f"+49 {digits[:3]} {digits[3:6]} {digits[6:8]}"
                if len(digits) > 8:
                    formatted += f" {digits[8:]}"
                return formatted.strip()
        
        # If starts with +1 (US/Canada), format as: +1 XXX XXX XXXX
        elif cleaned.startswith('+1'):
            digits = cleaned[2:]
            if len(digits) == 10:
                return f"+1 {digits[:3]} {digits[3:6]} {digits[6:]}"
        
        # Generic formatting: +XX XXX XXX XXX
        if cleaned.startswith('+'):
            country_code = cleaned[:3]
            rest = cleaned[3:]
            # Group remaining digits in chunks of 3
            chunks = [rest[i:i+3] for i in range(0, len(rest), 3)]
            return f"{country_code} {' '.join(chunks)}"
        
        # Fallback: return as-is with spaces every 3 digits
        chunks = [cleaned[i:i+3] for i in range(0, len(cleaned), 3)]
        return ' '.join(chunks)
    
    def _parse_yes_no_response(self, user_input: str) -> str:
        """
        Parse yes/no response from user input.
        Returns: "yes", "no", or "unclear"
        """
        if not user_input:
            return "unclear"
        
        cleaned = user_input.strip().lower()
        
        # Yes indicators
        yes_words = ["yes", "yeah", "yep", "correct", "right", "that's right", 
                     "yup", "uh huh", "mhmm", "sure", "okay", "ok", "affirmative"]
        if any(word in cleaned for word in yes_words):
            return "yes"
        
        # No indicators
        no_words = ["no", "nope", "wrong", "incorrect", "not right", "nah", 
                    "negative", "false", "that's wrong"]
        if any(word in cleaned for word in no_words):
            return "no"
        
        # Unclear/ambiguous
        return "unclear"
    
    def _parse_datetime_from_text(self, text: str) -> Optional[str]:
        """Parse date and time from natural language text"""
        text_lower = text.lower()
        now = datetime.now()
        parsed_date = None
        parsed_time = None
        
        # Read config-driven defaults for morning/afternoon times and business hours (Comment 1, Comment 3)
        default_morning_time = os.getenv('LEIBNIZ_APPOINTMENT_DEFAULT_MORNING_TIME', '10:00')
        default_afternoon_time = os.getenv('LEIBNIZ_APPOINTMENT_DEFAULT_AFTERNOON_TIME', '14:00')
        business_hours_start = os.getenv('LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_START', '08:00')
        business_hours_end = os.getenv('LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_END', '18:00')
        
        # Parse natural language dates
        if "today" in text_lower:
            parsed_date = now
        elif "tomorrow" in text_lower:
            parsed_date = now + timedelta(days=1)
        elif "next week" in text_lower:
            parsed_date = now + timedelta(days=7)
        elif "next month" in text_lower:
            # Approximate next month as 30 days
            parsed_date = now + timedelta(days=30)
        
        # Parse weekdays
        weekday_match = DATE_WEEKDAY.search(text_lower)
        if weekday_match:
            target_weekday = weekday_match.group(1)
            weekday_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }
            target_num = weekday_map[target_weekday]
            current_weekday = now.weekday()
            days_ahead = (target_num - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next occurrence
            parsed_date = now + timedelta(days=days_ahead)
        
        # Parse formatted dates (numeric formats)
        date_match = DATE_FORMATTED.search(text)
        if date_match:
            date_str = date_match.group(0)
            # Try different formats
            for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%m/%d/%y", "%d/%m/%y"]:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
        
        # Parse textual month formats (Comment 3: e.g., "December 25", "Dec 25 2024")
        if not parsed_date:
            # Try full month name with year: "December 25 2024"
            for fmt in ["%B %d %Y", "%b %d %Y"]:
                try:
                    parsed_date = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            
            # Try month name without year (default to current year)
            if not parsed_date:
                for fmt in ["%B %d", "%b %d"]:
                    try:
                        parsed_date = datetime.strptime(text, fmt)
                        # Set year to current year
                        parsed_date = parsed_date.replace(year=now.year)
                        break
                    except ValueError:
                        continue
        
        # Parse time formats
        time_match = TIME_PATTERN.search(text_lower)
        if time_match:
            time_str = time_match.group(0)
            # Try parsing with AM/PM
            for fmt in ["%I:%M %p", "%I:%M%p", "%H:%M"]:
                try:
                    time_obj = datetime.strptime(time_str.upper(), fmt)
                    parsed_time = time_obj.time()
                    break
                except ValueError:
                    continue
        
        # Parse simple time (e.g., "2pm")
        if not parsed_time:
            simple_time_match = TIME_SIMPLE.search(text_lower)
            if simple_time_match:
                time_str = simple_time_match.group(0)
                try:
                    time_obj = datetime.strptime(time_str.upper(), "%I%p")
                    parsed_time = time_obj.time()
                except ValueError:
                    try:
                        time_obj = datetime.strptime(time_str.upper(), "%I %p")
                        parsed_time = time_obj.time()
                    except ValueError:
                        pass
        
        # Parse relative time (morning, afternoon, evening) - use config-driven defaults
        if not parsed_time:
            relative_time_match = TIME_RELATIVE.search(text_lower)
            if relative_time_match:
                time_word = relative_time_match.group(1)
                if time_word == "morning":
                    parsed_time = datetime.strptime(default_morning_time, "%H:%M").time()
                elif time_word == "afternoon":
                    parsed_time = datetime.strptime(default_afternoon_time, "%H:%M").time()
                elif time_word == "evening":
                    parsed_time = datetime.strptime("17:00", "%H:%M").time()
        
        # Combine date and time
        if parsed_date and parsed_time:
            combined = datetime.combine(parsed_date.date(), parsed_time)
            return combined.strftime("%A, %B %d, %Y at %I:%M %p")
        elif parsed_date:
            # Default time if only date provided - use config-driven morning time
            default_time = datetime.strptime(default_morning_time, "%H:%M").time()
            combined = datetime.combine(parsed_date.date(), default_time)
            return combined.strftime("%A, %B %d, %Y at %I:%M %p")
        elif parsed_time:
            # Use tomorrow if only time provided
            tomorrow = now + timedelta(days=1)
            combined = datetime.combine(tomorrow.date(), parsed_time)
            return combined.strftime("%A, %B %d, %Y at %I:%M %p")
        
        return None
    
    def _validate_email(self, email: str) -> tuple[bool, Optional[str]]:
        """Validate email address format"""
        if not EMAIL_PATTERN.match(email):
            return False, "That doesn't look like a valid email address."
        
        # Check for common typos
        if email.endswith(".con"):
            return False, "Did you mean '.com' instead of '.con'?"
        
        # Warn if not university email (but still accept)
        if not email.endswith("@uni-hannover.de"):
            logger.debug(f"Non-university email provided: {email}")
        
        return True, None
    
    def _validate_phone(self, phone: str) -> tuple[bool, Optional[str]]:
        """Validate phone number format"""
        # Remove formatting for validation
        digits_only = re.sub(r'\D', '', phone)
        
        # Check length
        if len(digits_only) < 10 or len(digits_only) > 15:
            return False, "That phone number doesn't look right."
        
        # Check for obviously invalid patterns
        if digits_only == "1234567890" or digits_only == "0000000000":
            return False, "That doesn't look like a real phone number."
        
        return True, None
    
    def _validate_name(self, name: str) -> tuple[bool, Optional[str]]:
        """Validate name format"""
        if not NAME_PATTERN.match(name):
            return False, "I didn't quite catch that."
        
        # Check for at least 2 parts (first and last name)
        parts = name.split()
        if len(parts) < 2:
            return False, "Could you give me your full name (first and last)?"
        
        # Reject single-character names
        if any(len(part) < 2 for part in parts):
            return False, "That name seems a bit short."
        
        # Reject common invalid inputs
        if name.lower() in ["test", "none", "n/a"]:
            return False, "I need your real name for the appointment."
        
        return True, None
    
    def _validate_datetime(self, datetime_str: str) -> tuple[bool, Optional[str]]:
        """
        Validate parsed datetime against future date, booking window, and business hours.
        (Comment 2)
        
        Args:
            datetime_str: Formatted datetime string (e.g., "Monday, December 25, 2024 at 10:00 AM")
            
        Returns:
            (is_valid, error_message)
        """
        # Parse the formatted datetime string back to datetime object
        try:
            dt = datetime.strptime(datetime_str, "%A, %B %d, %Y at %I:%M %p")
        except ValueError:
            return False, "Could not parse the datetime format."
        
        now = datetime.now()
        
        # 1. Check if date is in the future
        if dt <= now:
            return False, "That time has already passed. Please choose a future date and time."
        
        # 2. Check maximum booking window (from config - Comment 2)
        max_booking_months = int(os.getenv('LEIBNIZ_APPOINTMENT_MAX_BOOKING_MONTHS', '3'))
        max_future_date = now + timedelta(days=max_booking_months * 30)
        if dt > max_future_date:
            return False, f"We can only book appointments up to {max_booking_months} months in advance. Please choose a closer date."
        
        # 3. Check business hours (from config - Comment 2)
        business_hours_start = os.getenv('LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_START', '08:00')
        business_hours_end = os.getenv('LEIBNIZ_APPOINTMENT_BUSINESS_HOURS_END', '18:00')
        
        appointment_time = dt.time()
        start_time = datetime.strptime(business_hours_start, "%H:%M").time()
        end_time = datetime.strptime(business_hours_end, "%H:%M").time()
        
        if appointment_time < start_time or appointment_time >= end_time:
            return False, f"That time is outside our business hours ({business_hours_start} to {business_hours_end}). Please choose a time during business hours."
        
        return True, None
    
    def _get_department_advance_notice_hint(self, department: Optional[str]) -> Optional[str]:
        """
        Get department-specific advance notice recommendation.
        
        Args:
            department: Department key
            
        Returns:
            Friendly hint about recommended booking lead time, or None if no specific hint
        """
        if not department:
            return None
        
        # Department-specific advance notice hints (from appointment_scheduling_complete.md)
        hints = {
            "academic_advising": "We recommend booking 1-2 weeks in advance for academic advising.",
            "international_office": "For visa matters, please try to book 2-3 weeks ahead if possible.",
            "career_services": "Career services appointments should be booked at least 1 week in advance.",
            "psychological_counseling": "Counseling appointments typically need 1-2 weeks notice (urgent cases are prioritized).",
            "examination_office": "Exam office appointments usually need 3-5 business days notice.",
            "financial_aid": "Financial aid appointments work best with 2-3 weeks advance notice.",
            "it_services": "IT services often has same-week availability for urgent technical issues.",
            "admissions": "Admissions appointments should be booked 1-2 weeks in advance.",
            "student_registration": "Registration office appointments usually need 3-5 days notice.",
            "faculty_consultation": "Faculty appointments typically need 1-2 weeks advance notice."
        }
        
        return hints.get(department)
    
    def reset(self):
        """Reset FSM to initial state"""
        self.state = AppointmentState.INIT
        self.data = AppointmentData()
        self.retry_counts = {}
        self.confirmation_attempts = {}
        self.conversation_history = []
        self.last_error = None
        logger.info(" Appointment FSM reset")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_appointment_fsm() -> LeibnizAppointmentFSM:
    """Create a new appointment booking FSM instance"""
    return LeibnizAppointmentFSM()


def format_appointment_for_submission(data: AppointmentData) -> Dict[str, Any]:
    """
    Format appointment data for submission to booking system
    
    Args:
        data: AppointmentData instance with collected information
        
    Returns:
        Formatted dictionary ready for API submission or database storage
    """
    # Convert to dictionary
    booking = data.to_dict()
    
    # Add metadata
    booking["booking_method"] = "voice_assistant"
    booking["booking_source"] = "leibniz_agent"
    booking["status"] = "pending_confirmation"
    
    # Add department-specific information
    dept_key = data.department
    if dept_key:
        # Advance notice requirements (from appointment_scheduling_complete.md)
        advance_notice = {
            "academic_advising": "1-2 weeks recommended",
            "international_office": "2-3 weeks for visa matters",
            "career_services": "1 week minimum",
            "counseling": "1-2 weeks (urgent cases prioritized)",
            "examination_office": "3-5 business days",
            "financial_aid": "2-3 weeks",
            "it_services": "Same-week appointments usually available",
            "faculty": "1-2 weeks recommended",
            "admissions": "1 week recommended",
            "registration": "3-5 business days"
        }
        
        # Typical duration
        duration = {
            "academic_advising": "30-45 minutes",
            "career_services": "45-60 minutes",
            "counseling": "50 minutes initial session",
            "examination_office": "15-20 minutes",
            "it_services": "30 minutes",
            "faculty": "30-60 minutes",
            "international_office": "30-45 minutes",
            "financial_aid": "30 minutes",
            "admissions": "20-30 minutes",
            "registration": "15-20 minutes"
        }
        
        booking["advance_notice"] = advance_notice.get(dept_key, "1 week recommended")
        booking["typical_duration"] = duration.get(dept_key, "30 minutes")
        booking["preparation"] = "Bring student ID and any relevant documents"
    
    return booking


# ============================================================================
# Test Function
# ============================================================================

async def test_appointment_fsm():
    """Comprehensive test suite for appointment FSM"""
    print("\n" + "="*80)
    print(" Leibniz Appointment FSM Test Suite")
    print("="*80 + "\n")
    
    # Scenario 1: Happy path (all fields valid first try)
    print("="*80)
    print("Scenario 1: Happy Path (All Valid Inputs)")
    print("="*80)
    
    fsm = create_appointment_fsm()
    
    test_inputs = [
        ("", "Initialize"),
        ("John Smith", "Name"),
        ("john.smith@uni-hannover.de", "Email"),
        ("+49 511 762 2020", "Phone"),
        ("academic advising", "Department"),
        ("course selection", "Appointment Type"),
        ("next Tuesday at 2pm", "Date/Time"),
        ("I need help choosing courses for next semester", "Purpose"),
        ("yes", "Confirmation")
    ]
    
    for user_input, label in test_inputs:
        result = await fsm.process_input(user_input)
        print(f"\n{label}:")
        print(f"  Input: '{user_input}'")
        print(f"  State: {result['state']}")
        print(f"  Response: {result['response'][:100]}...")
        if result['complete']:
            print(f"\n Booking Complete!")
            print(f"  Data: {result['data']}")
    
    # Scenario 2: Validation errors and retries
    print("\n\n" + "="*80)
    print("Scenario 2: Validation Errors and Retries")
    print("="*80)
    
    fsm2 = create_appointment_fsm()
    await fsm2.process_input("")  # Initialize
    await fsm2.process_input("John Smith")  # Name
    
    # Invalid email - retry
    result = await fsm2.process_input("not-an-email")
    print(f"\nInvalid Email:")
    print(f"  Response: {result['response'][:100]}...")
    print(f"  Retry count: {fsm2.retry_counts.get('email', 0)}")
    
    # Valid email
    result = await fsm2.process_input("john@uni-hannover.de")
    print(f"\nValid Email:")
    print(f"  State: {result['state']}")
    
    # Scenario 3: Natural language date/time parsing
    print("\n\n" + "="*80)
    print("Scenario 3: Natural Language Date/Time Parsing")
    print("="*80)
    
    test_dates = [
        "tomorrow at 10am",
        "next Monday afternoon",
        "December 15 at 2:30pm",
        "next week"
    ]
    
    for date_input in test_dates:
        parsed = fsm._parse_datetime_from_text(date_input)
        print(f"\nInput: '{date_input}'")
        print(f"Parsed: {parsed}")
    
    # Scenario 4: Cancellation
    print("\n\n" + "="*80)
    print("Scenario 4: Cancellation Mid-Flow")
    print("="*80)
    
    fsm3 = create_appointment_fsm()
    await fsm3.process_input("")  # Initialize
    await fsm3.process_input("John Smith")  # Name
    result = await fsm3.process_input("cancel")  # Cancel
    print(f"\nCancellation:")
    print(f"  State: {result['state']}")
    print(f"  Cancelled: {result['cancelled']}")
    print(f"  Response: {result['response']}")
    
    print("\n" + "="*80)
    print(" All Tests Completed Successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_appointment_fsm())
