# SeniorProject
Main system
    - Python
    - Lv5 AI -> Chain of thought
    - Output sound model (using eleven lap)
    - Using api from processing
    - Face interface to rect with sound

System
     ____________________________________________________________________________________________________________________________    
    |                                                                                                                            |
    | TTS (Text to speech) -> input -> api (Brain system process) -> processing -> STT (Speech to text) -> output -> sound model |
    |____________________________________________________________________________________________________________________________|

    - Have chain of though
    - Can agressive 
    - Face interface to rect with sound

Make AI Responses More Smooth and Natural
Overview
    Enhance Rose's responses to sound more natural and conversational by updating the system instruction with specific conversational guidelines and adding generation configuration parameters.

Changes
1. - Enhanced System Instruction (MainSystem.py lines 19-25)
    - Add specific guidelines for natural conversation:
    - Use shorter, simpler sentences that flow naturally
    - Vary sentence length for natural rhythm
    - Use casual, friendly language appropriate for a Thai friend
    - Avoid overly formal or structured responses
    - Sound like you're speaking, not writing a document
    - Keep responses concise but warm
2. - Generation Configuration (MainSystem.py around line 17)
    - Add generation_config parameter to the GenerativeModel:
    - Set temperature to ~0.8-0.9 for more natural variation
    - Add max_output_tokens if needed to encourage concise responses
    - Configure top_p/top_k for natural sampling
3. - Response Handling (MainSystem.py in ask function)
    - Consider minimal post-processing if needed, but prioritize letting the model generate naturally

Files to Modify
    - MainSystem.py: Update system instruction and add generation config