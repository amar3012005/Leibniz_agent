PS C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete> python leibniz_agent\test_end_to_end_flow.py
⚠️ ElevenLabs not available. Install with: pip install elevenlabs
C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\pygame\pkgdata.py:25: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import resource_stream, resource_exists
pygame 2.6.1 (SDL 2.28.4, Python 3.10.0)
Hello from the pygame community. https://www.pygame.org/contribute.html
2025-10-27 00:24:09,439 - __main__ - INFO - Starting Leibniz End-to-End Flow Tests
2025-10-27 00:24:09,439 - __main__ - INFO - 
============================================================
2025-10-27 00:24:09,439 - __main__ - INFO - Running test: Happy Path - Complete Flow    
2025-10-27 00:24:09,439 - __main__ - INFO - Description: Tests greeting → RAG query → appointment booking → exit
2025-10-27 00:24:09,439 - __main__ - INFO - ============================================================
2025-10-27 00:24:09,440 - __main__ - INFO - Initializing Leibniz services...
2025-10-27 00:24:09,440 - leibniz_agent.leibniz_persistent_services - INFO - 🎓 Initializing Leibniz Persistent Services Manager...
2025-10-27 00:24:09,440 - leibniz_agent.leibniz_persistent_services - INFO - 🎓 Initializing Leibniz Intent Parser...
🎓 Leibniz Parser: Gemini AI (gemini-2.0-flash-lite) enabled for complex classification 
🎓 Leibniz Intent Parser initialized (threshold=0.8, timeout=5.0s)
2025-10-27 00:24:09,440 - leibniz_agent.leibniz_persistent_services - INFO - ✅ Leibniz Intent Parser ready for university customer service
2025-10-27 00:24:09,442 - leibniz_agent.leibniz_persistent_services - INFO - 🎓 Initializing Leibniz RAG System...
2025-10-27 00:24:13,304 - sentence_transformers.SentenceTransformer - INFO - Load pretrained SentenceTransformer: sentence-transformers/all-MiniLM-L6-v2
✅ Embeddings model loaded successfully (sentence-transformers/all-MiniLM-L6-v2)
✅ Gemini 2.0 Flash initialized for Leibniz RAG (169.5 tok/s)
✅ Leibniz vector store loaded successfully (258 chunks)
🎓 Leibniz RAG: Initialized for university customer service
   Knowledge base: leibniz_knowledge_base
   Top-K: 8, Top-N: 5, Similarity threshold: 0.3
2025-10-27 00:24:16,828 - leibniz_agent.leibniz_persistent_services - INFO - ✅ Using existing Leibniz vector store (no rebuild needed)
2025-10-27 00:24:16,829 - leibniz_agent.leibniz_persistent_services - INFO - ✅ Leibniz RAG System ready for university queries!
2025-10-27 00:24:16,829 - leibniz_agent.leibniz_persistent_services - INFO - 📊 Vector store: 258 documents from 12 categories
2025-10-27 00:24:18,276 - leibniz_agent.leibniz_persistent_services - INFO - ✅ Verification successful - context-aware retrieval working
2025-10-27 00:24:18,276 - leibniz_agent.leibniz_persistent_services - INFO - 🔄 Intent parser processing loop started
2025-10-27 00:24:18,276 - leibniz_agent.leibniz_persistent_services - INFO - 🔄 RAG processing loop started
2025-10-27 00:24:18,277 - leibniz_agent.leibniz_persistent_services - INFO - ✅ Leibniz Persistent Services ready in 8.84s
2025-10-27 00:24:18,277 - leibniz_agent.leibniz_persistent_services - INFO - 📊 Intent parser ready, RAG system ready with 258 documents
2025-10-27 00:24:18,277 - __main__ - INFO -
Turn 1: User says 'Hello'
2025-10-27 00:24:18,278 - __main__ - INFO - Intent classified: GREETING
2025-10-27 00:24:18,278 - __main__ - INFO - Agent: Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?...
2025-10-27 00:24:18,278 - __main__ - INFO -
Turn 2: User says 'What are the requirements for the computer science program?'
2025-10-27 00:24:18,278 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:18,278 - __main__ - INFO - Agent: rag_1761521058278_2301105199616...   
2025-10-27 00:24:18,278 - __main__ - INFO -
Turn 3: User says 'I'd like to schedule an appointment with admissions'
2025-10-27 00:24:18,279 - __main__ - INFO - Intent classified: APPOINTMENT_SCHEDULING   
2025-10-27 00:24:18,279 - leibniz_agent.leibniz_appointment_fsm - INFO - 🎓 Leibniz Appointment FSM initialized
2025-10-27 00:24:18,279 - __main__ - INFO - Agent: Great! I'd be happy to help you schedule an appointment. I'll need to collect a few details from you...
2025-10-27 00:24:18,279 - __main__ - INFO -
Turn 4: User says 'John Smith'
2025-10-27 00:24:21,026 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 2.75s
2025-10-27 00:24:22,185 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:22,185 - __main__ - INFO - Agent: Thanks, John! Now, what's your email address? I'll send the appointment confirmation there....
2025-10-27 00:24:22,185 - __main__ - WARNING - Turn 4 took 3.91s (exceeds target of 2.5s)
2025-10-27 00:24:22,185 - __main__ - INFO -
Turn 5: User says 'john.smith@uni-hannover.de'
2025-10-27 00:24:23,255 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:23,256 - __main__ - INFO - Agent: Perfect! And what's your phone number? Include the country code if you're calling from outside Germa...
2025-10-27 00:24:23,257 - __main__ - INFO -
Turn 6: User says '+49 511 762 2020'
2025-10-27 00:24:24,272 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:24,272 - __main__ - INFO - Agent: Great! Now, which department would you like to schedule an appointment with? Here are your options:
...
2025-10-27 00:24:24,272 - __main__ - INFO -
Turn 7: User says 'admissions'
2025-10-27 00:24:24,272 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:24,272 - __main__ - INFO - Agent: rag_1761521064272_2301109231408...   
2025-10-27 00:24:24,273 - __main__ - INFO -
Turn 8: User says 'application questions'
2025-10-27 00:24:25,243 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 0.97s
2025-10-27 00:24:26,138 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:26,138 - __main__ - INFO - Agent: rag_1761521066138_2301109212128...
2025-10-27 00:24:26,138 - __main__ - INFO -
Turn 9: User says 'next Tuesday at 2pm'
2025-10-27 00:24:27,153 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.02s
2025-10-27 00:24:28,325 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:28,325 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:28,325 - __main__ - INFO -
Turn 10: User says 'I have questions about my application'
2025-10-27 00:24:29,338 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:29,340 - __main__ - INFO - Agent: rag_1761521069340_2301105027664...
2025-10-27 00:24:29,340 - __main__ - INFO -
Turn 11: User says 'yes'
2025-10-27 00:24:30,274 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 0.93s
2025-10-27 00:24:31,190 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:31,190 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:31,190 - __main__ - INFO -
Turn 12: User says 'Thanks, goodbye'
2025-10-27 00:24:31,190 - __main__ - INFO - Intent classified: EXIT
2025-10-27 00:24:31,191 - __main__ - INFO - Agent: Thanks for chatting! Have a great day, and feel free to reach out anytime you need help....
2025-10-27 00:24:31,191 - __main__ - INFO -
Running validation checks...
2025-10-27 00:24:31,191 - __main__ - INFO -   validate_all_intents_classified: FAIL     
2025-10-27 00:24:31,191 - __main__ - INFO -   validate_context_extracted: FAIL
2025-10-27 00:24:31,191 - __main__ - INFO -   validate_appointment_data_collected: FAIL 
2025-10-27 00:24:31,191 - __main__ - INFO -   validate_friendly_tone: PASS
2025-10-27 00:24:33,192 - __main__ - INFO - 
============================================================
2025-10-27 00:24:33,192 - __main__ - INFO - Running test: RAG-Only Flow
2025-10-27 00:24:33,192 - __main__ - INFO - Description: Tests multiple RAG queries without appointment booking
2025-10-27 00:24:33,192 - __main__ - INFO - ============================================================
2025-10-27 00:24:33,192 - __main__ - INFO - Initializing Leibniz services...
2025-10-27 00:24:33,192 - __main__ - INFO -
Turn 1: User says 'Hi'
2025-10-27 00:24:33,192 - __main__ - INFO - Intent classified: GREETING
2025-10-27 00:24:33,193 - __main__ - INFO - Agent: Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?...
2025-10-27 00:24:33,193 - __main__ - INFO -
Turn 2: User says 'Tell me about campus housing'
2025-10-27 00:24:33,193 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:33,193 - __main__ - INFO - Agent: rag_1761521073193_2301109213248...   
2025-10-27 00:24:33,193 - __main__ - INFO -
Turn 3: User says 'What about financial aid?'
2025-10-27 00:24:33,193 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:33,193 - __main__ - INFO - Agent: rag_1761521073193_2301109213328...   
2025-10-27 00:24:33,193 - __main__ - INFO -
Turn 4: User says 'How do I apply for scholarships?'
2025-10-27 00:24:33,193 - __main__ - INFO - Intent classified: GREETING
2025-10-27 00:24:33,193 - __main__ - INFO - Agent: Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?...
2025-10-27 00:24:33,193 - __main__ - INFO -
Turn 5: User says 'That's all, thanks'
2025-10-27 00:24:33,194 - __main__ - INFO - Intent classified: EXIT
2025-10-27 00:24:33,194 - __main__ - INFO - Agent: Thanks for chatting! Have a great day, and feel free to reach out anytime you need help....
2025-10-27 00:24:33,194 - __main__ - INFO -
Running validation checks...
2025-10-27 00:24:33,194 - __main__ - INFO -   validate_all_intents_classified: FAIL     
2025-10-27 00:24:33,194 - __main__ - INFO -   validate_multiple_rag_queries: FAIL       
2025-10-27 00:24:33,194 - __main__ - INFO -   validate_no_appointment_triggered: PASS   
2025-10-27 00:24:33,194 - __main__ - INFO -   validate_friendly_tone: PASS
2025-10-27 00:24:35,008 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.81s
2025-10-27 00:24:36,440 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.43s
2025-10-27 00:24:36,440 - __main__ - INFO - 
============================================================
2025-10-27 00:24:36,440 - __main__ - INFO - Running test: Appointment-Only Flow
2025-10-27 00:24:36,440 - __main__ - INFO - Description: Tests direct appointment booking without RAG queries
2025-10-27 00:24:36,440 - __main__ - INFO - ============================================================
2025-10-27 00:24:36,440 - __main__ - INFO - Initializing Leibniz services...
2025-10-27 00:24:36,440 - __main__ - INFO -
Turn 1: User says 'I need to book an appointment'
2025-10-27 00:24:36,440 - __main__ - INFO - Intent classified: APPOINTMENT_SCHEDULING   
2025-10-27 00:24:36,440 - leibniz_agent.leibniz_appointment_fsm - INFO - 🎓 Leibniz Appointment FSM initialized
2025-10-27 00:24:36,440 - __main__ - INFO - Agent: Great! I'd be happy to help you schedule an appointment. I'll need to collect a few details from you...
2025-10-27 00:24:36,440 - __main__ - INFO -
Turn 2: User says 'Emily Johnson'
2025-10-27 00:24:37,531 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:37,531 - __main__ - INFO - Agent: Thanks, Emily! Now, what's your email address? I'll send the appointment confirmation there....
2025-10-27 00:24:37,531 - __main__ - INFO -
Turn 3: User says 'emily.johnson@uni-hannover.de'
2025-10-27 00:24:38,791 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:38,792 - __main__ - INFO - Agent: Perfect! And what's your phone number? Include the country code if you're calling from outside Germa...
2025-10-27 00:24:38,792 - __main__ - INFO -
Turn 4: User says '+49 511 762 3030'
2025-10-27 00:24:39,932 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:39,932 - __main__ - INFO - Agent: Great! Now, which department would you like to schedule an appointment with? Here are your options:
...
2025-10-27 00:24:39,932 - __main__ - INFO -
Turn 5: User says 'career services'
2025-10-27 00:24:40,925 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:40,925 - __main__ - INFO - Agent: rag_1761521080925_2301109234992...
2025-10-27 00:24:40,925 - __main__ - INFO -
Turn 6: User says 'resume review'
2025-10-27 00:24:42,221 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.29s
2025-10-27 00:24:43,406 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:43,406 - __main__ - INFO - Agent: rag_1761521083406_2301109235056...
2025-10-27 00:24:43,406 - __main__ - INFO -
Turn 7: User says 'tomorrow at 10am'
2025-10-27 00:24:44,405 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.00s
2025-10-27 00:24:45,394 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:45,394 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:45,394 - __main__ - INFO -
Turn 8: User says 'I need help updating my resume for internships'
2025-10-27 00:24:45,394 - __main__ - INFO - Intent classified: GREETING
2025-10-27 00:24:45,394 - __main__ - INFO - Agent: Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?...
2025-10-27 00:24:45,394 - __main__ - INFO -
Turn 9: User says 'yes'
2025-10-27 00:24:46,303 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:46,303 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:46,303 - __main__ - INFO -
Turn 10: User says 'Goodbye'
2025-10-27 00:24:46,303 - __main__ - INFO - Intent classified: EXIT
2025-10-27 00:24:46,304 - __main__ - INFO - Agent: Thanks for chatting! Have a great day, and feel free to reach out anytime you need help....
2025-10-27 00:24:46,304 - __main__ - INFO -
Running validation checks...
2025-10-27 00:24:46,304 - __main__ - INFO -   validate_direct_appointment_booking: PASS 
2025-10-27 00:24:46,304 - __main__ - INFO -   validate_appointment_data_collected: FAIL 
2025-10-27 00:24:46,304 - __main__ - INFO -   validate_friendly_tone: PASS
2025-10-27 00:24:48,304 - __main__ - INFO - 
============================================================
2025-10-27 00:24:48,304 - __main__ - INFO - Running test: Mixed Flow with Interruptions
2025-10-27 00:24:48,304 - __main__ - INFO - Description: Tests transitions between RAG and appointment modes
2025-10-27 00:24:48,305 - __main__ - INFO - ============================================================
2025-10-27 00:24:48,305 - __main__ - INFO - Initializing Leibniz services...
2025-10-27 00:24:48,305 - __main__ - INFO -
Turn 1: User says 'Hello'
2025-10-27 00:24:48,306 - __main__ - INFO - Intent classified: GREETING
2025-10-27 00:24:48,306 - __main__ - INFO - Agent: Hi there! I'm Lexi, your friendly assistant for Leibniz University. How can I help you today?...
2025-10-27 00:24:48,306 - __main__ - INFO -
Turn 2: User says 'What are the CS program requirements?'
2025-10-27 00:24:48,306 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:48,307 - __main__ - INFO - Agent: rag_1761521088307_2301105236528...   
2025-10-27 00:24:48,307 - __main__ - INFO -
Turn 3: User says 'Actually, I want to schedule an appointment'
2025-10-27 00:24:48,307 - __main__ - INFO - Intent classified: APPOINTMENT_SCHEDULING   
2025-10-27 00:24:48,307 - leibniz_agent.leibniz_appointment_fsm - INFO - 🎓 Leibniz Appointment FSM initialized
2025-10-27 00:24:48,308 - __main__ - INFO - Agent: Great! I'd be happy to help you schedule an appointment. I'll need to collect a few details from you...
2025-10-27 00:24:48,308 - __main__ - INFO -
Turn 4: User says 'Sarah Williams'
2025-10-27 00:24:50,895 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 2.59s
2025-10-27 00:24:51,798 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:51,798 - __main__ - INFO - Agent: Thanks, Sarah! Now, what's your email address? I'll send the appointment confirmation there....
2025-10-27 00:24:51,798 - __main__ - WARNING - Turn 4 took 3.49s (exceeds target of 2.5s)
2025-10-27 00:24:51,798 - __main__ - INFO -
Turn 5: User says 'sarah.williams@uni-hannover.de'
2025-10-27 00:24:52,754 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:52,754 - __main__ - INFO - Agent: Perfect! And what's your phone number? Include the country code if you're calling from outside Germa...
2025-10-27 00:24:52,754 - __main__ - INFO -
Turn 6: User says '+49 511 762 4040'
2025-10-27 00:24:53,654 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:53,654 - __main__ - INFO - Agent: Great! Now, which department would you like to schedule an appointment with? Here are your options:
...
2025-10-27 00:24:53,654 - __main__ - INFO -
Turn 7: User says 'academic advising'
2025-10-27 00:24:54,642 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:54,643 - __main__ - INFO - Agent: rag_1761521094643_2301109215568...
2025-10-27 00:24:54,643 - __main__ - INFO -
Turn 8: User says 'course selection'
2025-10-27 00:24:54,643 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:54,643 - __main__ - INFO - Agent: rag_1761521094643_2301109215648...   
2025-10-27 00:24:54,643 - __main__ - INFO -
Turn 9: User says 'Friday at 3pm'
2025-10-27 00:24:56,268 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.62s
2025-10-27 00:24:57,536 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 1.27s
2025-10-27 00:24:58,567 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:58,567 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:58,567 - __main__ - WARNING - Turn 9 took 3.92s (exceeds target of 2.5s)
2025-10-27 00:24:58,567 - __main__ - INFO -
Turn 10: User says 'I need help choosing my courses for next semester'
2025-10-27 00:24:58,568 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:58,568 - __main__ - INFO - Agent: rag_1761521098568_2301105194912...   
2025-10-27 00:24:58,568 - __main__ - INFO -
Turn 11: User says 'yes'
❌ Error processing RAG query: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit.
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 30
Please retry in 1.148842003s. [violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests" 
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash-lite"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
  quota_value: 30
}
, links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, retry_delay {
  seconds: 1
}
]
Traceback (most recent call last):
  File "C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\leibniz_agent\leibniz_rag.py", line 784, in process_rag_query
    response = self.gemini_model.generate_content(prompt)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\generativeai\generative_models.py", line 331, in generate_content
    response = self._client.generate_content(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\ai\generativelanguage_v1beta\services\generative_service\client.py", line 835, in generate_content
    response = rpc(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\gapic_v1\method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 147, in retry_target
    result = target()
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\timeout.py", line 130, in func_with_timeout
    return func(*args, **kwargs)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\grpc_helpers.py", line 78, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exc
google.api_core.exceptions.ResourceExhausted: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit.
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 30
Please retry in 1.148842003s. [violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests" 
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash-lite"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
  quota_value: 30
}
, links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, retry_delay {
  seconds: 1
}
]
2025-10-27 00:24:58,702 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 0.13s
⚠️ Gemini API error: 429 You exceeded your current quota, please check your plan and billling details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit.
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 30
Please retry in 1.054764s. [violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests" 
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash-lite"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
  quota_value: 30
}
, links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, retry_delay {
  seconds: 1
}
]
2025-10-27 00:24:58,788 - __main__ - INFO - Intent classified: UNCLEAR
2025-10-27 00:24:58,788 - __main__ - INFO - Agent: I didn't catch which department you need. Could you choose from this list?

1. Academic Advising
2. ...
2025-10-27 00:24:58,789 - __main__ - INFO -
Turn 12: User says 'Can you tell me about the library?'
2025-10-27 00:24:58,789 - __main__ - INFO - Intent classified: RAG_QUERY
2025-10-27 00:24:58,789 - __main__ - INFO - Agent: rag_1761521098789_2301105236816...   
2025-10-27 00:24:58,789 - __main__ - INFO -
Turn 13: User says 'Bye'
2025-10-27 00:24:58,789 - __main__ - INFO - Intent classified: EXIT
2025-10-27 00:24:58,789 - __main__ - INFO - Agent: Thanks for chatting! Have a great day, and feel free to reach out anytime you need help....
2025-10-27 00:24:58,789 - __main__ - INFO -
Running validation checks...
2025-10-27 00:24:58,789 - __main__ - INFO -   validate_smooth_transitions: PASS
2025-10-27 00:24:58,789 - __main__ - INFO -   validate_context_maintained: PASS
2025-10-27 00:24:58,789 - __main__ - INFO -   validate_appointment_data_collected: FAIL 
2025-10-27 00:24:58,789 - __main__ - INFO -   validate_friendly_tone: PASS
❌ Error processing RAG query: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit.
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 30
Please retry in 916.164522ms. [violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests" 
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash-lite"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
  quota_value: 30
}
, links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, retry_delay {
}
]
Traceback (most recent call last):
  File "C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete\leibniz_agent\leibniz_rag.py", line 784, in process_rag_query
    response = self.gemini_model.generate_content(prompt)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\generativeai\generative_models.py", line 331, in generate_content
    response = self._client.generate_content(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\ai\generativelanguage_v1beta\services\generative_service\client.py", line 835, in generate_content
    response = rpc(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\gapic_v1\method.py", line 131, in __call__
    return wrapped_func(*args, **kwargs)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 294, in retry_wrapped_func
    return retry_target(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 156, in retry_target
    next_sleep = _retry_error_helper(
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_base.py", line 214, in _retry_error_helper
    raise final_exc from source_exc
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\retry\retry_unary.py", line 147, in retry_target
    result = target()
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\timeout.py", line 130, in func_with_timeout
    return func(*args, **kwargs)
  File "C:\Users\AMAR\AppData\Local\Programs\Python\Python310\lib\site-packages\google\api_core\grpc_helpers.py", line 78, in error_remapped_callable
    raise exceptions.from_grpc_error(exc) from exc
google.api_core.exceptions.ResourceExhausted: 429 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/usage?tab=rate-limit.
* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 30
Please retry in 916.164522ms. [violations {
  quota_metric: "generativelanguage.googleapis.com/generate_content_free_tier_requests" 
  quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
  quota_dimensions {
    key: "model"
    value: "gemini-2.0-flash-lite"
  }
  quota_dimensions {
    key: "location"
    value: "global"
  }
  quota_value: 30
}
, links {
  description: "Learn more about Gemini API quotas"
  url: "https://ai.google.dev/gemini-api/docs/rate-limits"
}
, retry_delay {
}
]
2025-10-27 00:24:58,921 - leibniz_agent.leibniz_persistent_services - INFO - ⚡ RAG query processed in 0.13s

================================================================================        
END-TO-END FLOW TEST RESULTS
================================================================================        

Test Execution Summary:
  Total Scenarios: 4
  Passed: 0 ✅
  Failed: 4 ❌
  Errors: 0 🔧
  Total Duration: 42.10s

Performance Metrics:
  Average Response Time: 0.83s
  Target Met: Yes ✅

Issues Found (4):
  - [HIGH] Happy Path - Complete Flow: Scenario failed with status FAIL
  - [HIGH] RAG-Only Flow: Scenario failed with status FAIL
  - [HIGH] Appointment-Only Flow: Scenario failed with status FAIL
  - [HIGH] Mixed Flow with Interruptions: Scenario failed with status FAIL

Recommendations:
  - [CRITICAL] Fix failing scenarios - 4 scenarios are not passing validation

================================================================================        
PS C:\Users\AMAR\SINDHv2\SINDH-Orchestra-Complete> 