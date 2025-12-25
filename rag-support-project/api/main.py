# api/main.py
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import Form
from dotenv import load_dotenv
import os
import json
import logging
from api.models import ChatRequest, User
from api.services.embeddings import get_embedding
from api.services.vector_store import search_vectors
from api.services.llm import stream_llm_response, get_llm_response
from api.middleware.auth import get_current_user
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from urllib.parse import unquote, parse_qs
import logging
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


def dev_mode_enabled() -> bool:
    return os.getenv("DEV_MODE", "true").lower() in ("1", "true", "yes")


@app.get("/api/health")
async def health_check():
    """Simple health endpoint that attempts to detect whether optional services are available."""
    checks = {
        "dev_mode": dev_mode_enabled(),
        "qdrant_url": bool(os.getenv("QDRANT_URL")),
        "groq_api_key": bool(os.getenv("GROQ_API_KEY")),
    }
    return {"status": "ok", "checks": checks}


@app.on_event("startup")
async def _startup_checks():
    # Log basic expectations and environment status. Do not import heavy libs at
    # module import time; attempt to import them in a try/except and warn.
    logger.info("Starting API; DEV_MODE=%s", dev_mode_enabled())
    if not dev_mode_enabled():
        if not os.getenv("QDRANT_URL"):
            logger.warning("QDRANT_URL is not set. Vector search will be unavailable.")
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY is not set. LLM streaming will be unavailable.")

    # Quick try-import checks to surface potential install issues early
    for pkg in ("sentence_transformers", "qdrant_client", "groq"):
        try:
            __import__(pkg)
            logger.info("%s import OK", pkg)
        except Exception as e:
            logger.warning("Unable to import %s: %s", pkg, str(e))


# @app.post("/api/chat")
# async def chat(
#         request: Request,
#         user: User = Depends(get_current_user),
#         format: str = Query("stream", description="Response format: 'stream' for SSE or 'json' for JSON")
# ):
#     # Handle both JSON and form data
#     content_type = request.headers.get("content-type", "").lower()
#     query = None
    
#     try:
#         if "application/json" in content_type:
#             # JSON request
#             body = await request.json()
#             query = body.get("query", "")
#         elif "application/x-www-form-urlencoded" in content_type:
#             # URL-encoded form data
#             body_bytes = await request.body()
#             body_str = body_bytes.decode("utf-8")
            
#             # Try to parse as URL-encoded form first
#             from urllib.parse import unquote, parse_qs
#             try:
#                 # Decode URL encoding
#                 decoded = unquote(body_str)
#                 # Try parsing as form data
#                 parsed = parse_qs(decoded)
#                 query = parsed.get("query", [None])[0]
                
#                 # If not found, try parsing the entire decoded string as JSON
#                 if not query:
#                     try:
#                         # Remove trailing '=' if present
#                         decoded_clean = decoded.rstrip('=')
#                         body_json = json.loads(decoded_clean)
#                         query = body_json.get("query", "")
#                     except:
#                         pass
#             except Exception as e:
#                 # If form parsing fails, try parsing as JSON directly
#                 try:
#                     decoded = unquote(body_str)
#                     decoded_clean = decoded.rstrip('=')
#                     body_json = json.loads(decoded_clean)
#                     query = body_json.get("query", "")
#                 except:
#                     pass
#         else:
#             # Try to parse as JSON anyway
#             try:
#                 body = await request.json()
#                 query = body.get("query", "")
#             except:
#                 # Last resort: try reading raw body and parsing
#                 try:
#                     body_bytes = await request.body()
#                     body_str = body_bytes.decode("utf-8")
#                     from urllib.parse import unquote
#                     decoded = unquote(body_str).rstrip('=')
#                     body_json = json.loads(decoded)
#                     query = body_json.get("query", "")
#                 except:
#                     pass
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")
    
#     if not query:
#         raise HTTPException(status_code=400, detail="Missing 'query' field in request body. Expected JSON: {\"query\": \"your question\"}")
#     # If DEV_MODE is enabled, return a canned response and skip heavy external services
#     if dev_mode_enabled():
#         async def _canned_gen():
#             yield "data: This is a dev environment fallback response.\n\n"
#         return StreamingResponse(_canned_gen(), media_type="text/event-stream")

#     try:
#         # 1. Embed query
#         try:
#             query_vector = await get_embedding(query)
#         except RuntimeError as e:
#             logger.error("Embedding initialization failed: %s", e)
#             raise HTTPException(status_code=503, detail=str(e))

#         # 2. Retrieve context
#         try:
#             results = await search_vectors(query_vector, top_k=5, threshold=0.7)
#         except RuntimeError as e:
#             logger.error("Vector search failed: %s", e)
#             raise HTTPException(status_code=503, detail=str(e))

#         # If retrieval returns no results, send a graceful fallback message instead of erroring
#         if not results:
#             fallback_msg = (
#                 "I couldn't find any relevant information in the knowledge base for your question. "
#                 "Please try rephrasing your query or provide more details."
#             )
#             if format.lower() == "json":
#                 return JSONResponse(
#                     content={
#                         "response": fallback_msg,
#                         "query": query,
#                         "sources": [],
#                     }
#                 )

#             async def _fallback_stream():
#                 yield f"data: {fallback_msg}\n\n"

#             return StreamingResponse(_fallback_stream(), media_type="text/event-stream")

#         context = "\n\n".join([r.payload["text"] for r in results])

#         # 3. Build prompt
#         prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

#         # 4. Return response based on format
#         if format.lower() == "json":
#             # Return JSON response
#             try:
#                 response_text = await get_llm_response(prompt)
#                 return JSONResponse(
#                     content={
#                         "response": response_text,
#                         "query": query,
#                         "sources": [
#                             {
#                                 "source": r.payload.get("source", "unknown"),
#                                 "score": r.score,
#                             }
#                             for r in results
#                         ],
#                     }
#                 )
#             except RuntimeError as e:
#                 logger.error("LLM initialization failed: %s", e)
#                 raise HTTPException(status_code=503, detail=str(e))
#         else:
#             # Return streaming response (default)
#             try:
#                 stream = stream_llm_response(prompt)
#             except RuntimeError as e:
#                 logger.error("LLM initialization failed: %s", e)
#                 raise HTTPException(status_code=503, detail=str(e))
#             return StreamingResponse(stream, media_type="text/event-stream")
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.exception("Unhandled exception in /api/chat: %s", e)
#         raise HTTPException(500, str(e))



# ============================================================================
# USER ENGAGEMENT TRACKING MODELS
# ============================================================================

class ConversationMetrics:
    """Track conversation-level engagement metrics"""
    def __init__(self):
        self.session_start = datetime.utcnow()
        self.message_count = 0
        self.total_response_time = 0.0
        self.user_wait_times = []
        self.abandonment_points = []
        self.re_engagement_attempts = 0
        self.context_switches = 0
        self.clarification_requests = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_duration": (datetime.utcnow() - self.session_start).total_seconds(),
            "message_count": self.message_count,
            "avg_response_time": self.total_response_time / max(self.message_count, 1),
            "avg_user_wait_time": sum(self.user_wait_times) / len(self.user_wait_times) if self.user_wait_times else 0,
            "re_engagement_attempts": self.re_engagement_attempts,
            "context_switches": self.context_switches,
            "clarification_requests": self.clarification_requests
        }

class UserEngagementProfile:
    """Profile for tracking user behavior and preferences"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.first_seen = datetime.utcnow()
        self.last_seen = datetime.utcnow()
        self.total_sessions = 1
        self.total_messages = 0
        self.preferred_response_style = "balanced"  # concise, balanced, detailed
        self.topics_of_interest = []
        self.average_session_length = 0.0
        self.successful_resolutions = 0
        self.frustration_indicators = 0
        
    def update_visit(self):
        self.last_seen = datetime.utcnow()
        self.total_sessions += 1
        
    def is_returning_user(self) -> bool:
        return self.total_sessions > 1
        
    def calculate_engagement_score(self) -> float:
        """Calculate overall engagement score (0-100)"""
        score = 0.0
        # Frequency component
        if self.total_sessions > 10:
            score += 30
        elif self.total_sessions > 5:
            score += 20
        elif self.total_sessions > 1:
            score += 10
            
        # Recency component
        days_since_last = (datetime.utcnow() - self.last_seen).days
        if days_since_last < 7:
            score += 30
        elif days_since_last < 30:
            score += 20
        elif days_since_last < 90:
            score += 10
            
        # Success rate component
        if self.total_messages > 0:
            success_rate = self.successful_resolutions / self.total_messages
            score += success_rate * 40
            
        return min(score, 100.0)

# ============================================================================
# ENGAGEMENT ENHANCEMENT FUNCTIONS
# ============================================================================

def detect_entry_context(request: Request) -> Dict[str, Any]:
    """Detect how user arrived at the chatbot"""
    context = {
        "entry_point": "direct",
        "referrer": None,
        "search_query": None,
        "user_agent": None,
        "is_mobile": False
    }
    
    # Detect referrer
    referrer = request.headers.get("referer")
    if referrer:
        context["referrer"] = referrer
        if "google" in referrer or "bing" in referrer:
            context["entry_point"] = "search"
        elif "facebook" in referrer or "twitter" in referrer or "linkedin" in referrer:
            context["entry_point"] = "social"
        else:
            context["entry_point"] = "referral"
    
    # Detect device type
    user_agent = request.headers.get("user-agent", "").lower()
    context["user_agent"] = user_agent
    context["is_mobile"] = any(device in user_agent for device in ["mobile", "android", "iphone"])
    
    # Extract search query if available
    query_params = dict(request.query_params)
    if "q" in query_params or "search" in query_params:
        context["search_query"] = query_params.get("q") or query_params.get("search")
    
    return context

def generate_personalized_greeting(
    profile: Optional[UserEngagementProfile],
    entry_context: Dict[str, Any]
) -> str:
    """Generate contextual greeting based on user profile and entry point"""
    
    if profile and profile.is_returning_user():
        return (
            f"Welcome back! I see you've visited {profile.total_sessions} times. "
            "How can I help you today?"
        )
    
    # First-time visitor greetings based on entry point
    greetings = {
        # Use plain-language, friendly greetings (avoid technical or product terms)
        "search": "Hi! Thanks for stopping by — how can I help you today?",
        "social": "Hello! Thanks for visiting — what can I help you with?",
        "referral": "Welcome! How can I help you today?",
        "direct": "Hello! I'm here to help — what's on your mind?"
    }
    
    base_greeting = greetings.get(entry_context.get("entry_point", "direct"))
    
    # Add mobile-specific guidance
    if entry_context.get("is_mobile"):
        base_greeting += " Feel free to keep your questions concise for easier mobile reading."
    
    return base_greeting

def assess_query_clarity(query: str) -> Dict[str, Any]:
    """Analyze query clarity and complexity"""
    words = query.split()
    
    assessment = {
        "length": len(words),
        "has_question_mark": "?" in query,
        "is_greeting": any(word in query.lower() for word in ["hi", "hello", "hey", "greetings"]),
        "is_vague": len(words) < 3 and not query.strip().endswith("?"),
        "complexity": "simple" if len(words) < 10 else "moderate" if len(words) < 20 else "complex",
        "needs_clarification": False
    }
    
    # Detect vague queries
    vague_patterns = ["it", "that", "this", "thing", "stuff"]
    if any(pattern in query.lower().split() for pattern in vague_patterns) and len(words) < 8:
        assessment["needs_clarification"] = True
    
    return assessment

def generate_clarification_prompt(query: str, assessment: Dict[str, Any]) -> Optional[str]:
    """Generate helpful clarification prompt if needed"""
    
    if assessment["is_greeting"]:
        # Use a short, friendly, and non-technical greeting
        return (
            "Hello! I'm here to help. What can I help you with today?"
        )
    
    if assessment["is_vague"]:
        # Keep clarification requests simple and user-friendly (avoid technical terms)
        return (
            "I'd love to help — could you share a little more detail about what you need? "
            "A short example or a couple more words would be great."
        )
    
    if assessment["needs_clarification"]:
        # Ask for rephrasing in plain language
        return (
            "Could you say that another way or add one or two details so I can help better?"
        )
    
    return None

def calculate_cognitive_load(context: str, response: str) -> Dict[str, Any]:
    """Assess cognitive load of the response"""
    
    response_words = response.split()
    context_words = context.split()
    
    load_metrics = {
        "response_length": len(response_words),
        "information_density": len(set(response_words)) / len(response_words) if response_words else 0,
        "context_ratio": len(context_words) / max(len(response_words), 1),
        "readability": "high" if len(response_words) < 100 else "moderate" if len(response_words) < 250 else "low",
        "recommended_chunk": len(response_words) > 200
    }
    
    return load_metrics

def detect_frustration_indicators(query: str, message_count: int) -> bool:
    """Detect signs of user frustration"""
    
    frustration_signals = [
        "not working", "doesn't work", "broken", "error", "wrong",
        "confused", "don't understand", "unclear", "help",
        "again", "still", "yet", "why"
    ]
    
    # Check for frustration keywords
    has_frustration_keywords = any(signal in query.lower() for signal in frustration_signals)
    
    # Check for repeated similar queries (simplified check)
    is_repetitive = message_count > 3 and len(query.split()) < 5
    
    # Check for excessive punctuation
    has_excessive_punctuation = query.count("!") > 2 or query.count("?") > 2
    
    return has_frustration_keywords or is_repetitive or has_excessive_punctuation

def generate_recovery_message(frustration_level: str) -> str:
    """Generate empathetic recovery message"""
    
    messages = {
        "low": "Let me try to explain that differently.",
        "moderate": "I understand this might be confusing. Let me break it down step by step.",
        "high": "I apologize if my previous responses weren't helpful. Let me start fresh - could you tell me exactly what you're trying to accomplish?"
    }
    
    return messages.get(frustration_level, messages["moderate"])

# ============================================================================
# SESSION STORAGE (In production, use Redis or database)
# ============================================================================

# In-memory storage for demo (replace with persistent storage)
user_profiles: Dict[str, UserEngagementProfile] = {}
conversation_metrics: Dict[str, ConversationMetrics] = {}

def get_user_profile(user_id: str) -> UserEngagementProfile:
    """Get or create user profile"""
    if user_id not in user_profiles:
        user_profiles[user_id] = UserEngagementProfile(user_id)
    else:
        user_profiles[user_id].update_visit()
    return user_profiles[user_id]

def get_conversation_metrics(session_id: str) -> ConversationMetrics:
    """Get or create conversation metrics"""
    if session_id not in conversation_metrics:
        conversation_metrics[session_id] = ConversationMetrics()
    return conversation_metrics[session_id]

# ============================================================================
# ENHANCED CHAT ENDPOINT
# ============================================================================

@app.post("/api/chat")
async def chat(
    request: Request,
    user: User = Depends(get_current_user),
    format: str = Query("stream", description="Response format: 'stream' for SSE or 'json' for JSON"),
    session_id: Optional[str] = Query(None, description="Session ID for tracking")
):
    start_time = time.time()
    
    # Generate session ID if not provided
    if not session_id:
        session_id = f"{user.id}_{int(time.time())}"
    
    # Initialize engagement tracking
    entry_context = detect_entry_context(request)
    user_profile = get_user_profile(user.id)
    conv_metrics = get_conversation_metrics(session_id)
    
    # Handle both JSON and form data
    content_type = request.headers.get("content-type", "").lower()
    query = None
    
    try:
        if "application/json" in content_type:
            body = await request.json()
            query = body.get("query", "")
        elif "application/x-www-form-urlencoded" in content_type:
            body_bytes = await request.body()
            body_str = body_bytes.decode("utf-8")
            
            from urllib.parse import unquote, parse_qs
            try:
                decoded = unquote(body_str)
                parsed = parse_qs(decoded)
                query = parsed.get("query", [None])[0]
                
                if not query:
                    try:
                        decoded_clean = decoded.rstrip('=')
                        body_json = json.loads(decoded_clean)
                        query = body_json.get("query", "")
                    except:
                        pass
            except Exception as e:
                try:
                    decoded = unquote(body_str)
                    decoded_clean = decoded.rstrip('=')
                    body_json = json.loads(decoded_clean)
                    query = body_json.get("query", "")
                except:
                    pass
        else:
            try:
                body = await request.json()
                query = body.get("query", "")
            except:
                try:
                    body_bytes = await request.body()
                    body_str = body_bytes.decode("utf-8")
                    decoded = unquote(body_str).rstrip('=')
                    body_json = json.loads(decoded)
                    query = body_json.get("query", "")
                except:
                    pass
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing request: {str(e)}")
    
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field in request body")
    
    # Update metrics
    conv_metrics.message_count += 1
    user_profile.total_messages += 1
    
    # Assess query clarity
    query_assessment = assess_query_clarity(query)
    
    # Check for first message in conversation
    if conv_metrics.message_count == 1:
        greeting = generate_personalized_greeting(user_profile, entry_context)
        # Could prepend greeting to response or send separately
    
    # Check for frustration
    is_frustrated = detect_frustration_indicators(query, conv_metrics.message_count)
    if is_frustrated:
        conv_metrics.re_engagement_attempts += 1
        user_profile.frustration_indicators += 1
    
    # Generate clarification if needed
    clarification = generate_clarification_prompt(query, query_assessment)
    if clarification and conv_metrics.message_count == 1:
        # For first unclear message, provide guidance
        if format.lower() == "json":
            return JSONResponse(content={
                "response": clarification,
                "query": query,
                "needs_clarification": True,
                "engagement_metrics": conv_metrics.to_dict()
            })
        
        async def _clarification_stream():
            # Emit plain text in the SSE stream (clients expect the raw message text)
            yield f"data: {clarification}\n\n"
        
        return StreamingResponse(_clarification_stream(), media_type="text/event-stream")
    
    # Dev mode check
    if dev_mode_enabled():
        async def _canned_gen():
            yield "data: This is a dev environment fallback response.\n\n"
        return StreamingResponse(_canned_gen(), media_type="text/event-stream")

    try:
        # 1. Embed query
        try:
            query_vector = await get_embedding(query)
        except RuntimeError as e:
            logger.error("Embedding initialization failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))

        # 2. Retrieve context
        try:
            results = await search_vectors(query_vector, top_k=5, threshold=0.7)
        except RuntimeError as e:
            logger.error("Vector search failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e))

        # Handle no results with empathetic fallback
        if not results:
            conv_metrics.clarification_requests += 1
            
            fallback_msg = (
                "I couldn't find specific information about that in my knowledge base. "
            )
            
            if is_frustrated:
                fallback_msg = generate_recovery_message("moderate") + " " + fallback_msg
            
            fallback_msg += (
                "Could you try rephrasing your question or asking about a related topic? "
                "I'm here to help!"
            )
            
            if format.lower() == "json":
                return JSONResponse(content={
                    "response": fallback_msg,
                    "query": query,
                    "sources": [],
                    "engagement_metrics": conv_metrics.to_dict(),
                    "no_results": True
                })

            async def _fallback_stream():
                yield f"data: {json.dumps({'text': fallback_msg, 'no_results': True})}\n\n"

            return StreamingResponse(_fallback_stream(), media_type="text/event-stream")

        context = "\n\n".join([r.payload["text"] for r in results])
        
        # Calculate cognitive load
        load_metrics = calculate_cognitive_load(context, query)

        # 3. Build enhanced prompt with engagement considerations
        prompt = f"Context:\n{context}\n\n"
        
        if is_frustrated:
            prompt += generate_recovery_message("moderate") + "\n\n"
        
        if user_profile.preferred_response_style == "concise":
            prompt += "Provide a concise, direct answer.\n\n"
        elif user_profile.preferred_response_style == "detailed":
            prompt += "Provide a comprehensive, detailed answer.\n\n"
        
        prompt += f"Question: {query}\n\nAnswer:"

        # 4. Track response time
        response_start = time.time()

        # 5. Return response based on format
        if format.lower() == "json":
            try:
                response_text = await get_llm_response(prompt)
                
                response_time = time.time() - response_start
                conv_metrics.total_response_time += response_time
                
                # Mark as successful if response generated
                user_profile.successful_resolutions += 1
                
                return JSONResponse(content={
                    "response": response_text,
                    "query": query,
                    "sources": [
                        {
                            "source": r.payload.get("source", "unknown"),
                            "score": r.score,
                        }
                        for r in results
                    ],
                    "engagement_metrics": conv_metrics.to_dict(),
                    "cognitive_load": load_metrics,
                    "response_time": response_time,
                    "session_id": session_id
                })
            except RuntimeError as e:
                logger.error("LLM initialization failed: %s", e)
                raise HTTPException(status_code=503, detail=str(e))
        else:
            # Streaming response with engagement metadata
            try:
                async def enhanced_stream():
                    """Stream with engagement tracking"""
                    response_start = time.time()
                    token_count = 0
                    
                    async for chunk in stream_llm_response(prompt):
                        token_count += 1
                        yield chunk
                    
                    # Send final engagement metrics
                    response_time = time.time() - response_start
                    conv_metrics.total_response_time += response_time
                    user_profile.successful_resolutions += 1
                    
                    final_metrics = {
                        "type": "metrics",
                        "engagement_metrics": conv_metrics.to_dict(),
                        "response_time": response_time,
                        "tokens_streamed": token_count,
                        "session_id": session_id
                    }
                    
                    yield f"data: {json.dumps(final_metrics)}\n\n"
                
                return StreamingResponse(enhanced_stream(), media_type="text/event-stream")
            except RuntimeError as e:
                logger.error("LLM initialization failed: %s", e)
                raise HTTPException(status_code=503, detail=str(e))
                
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in /api/chat: %s", e)
        raise HTTPException(500, str(e))
    finally:
        # Log total request time
        total_time = time.time() - start_time
        logger.info(f"Request completed in {total_time:.2f}s for user {user.id}")

# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/api/engagement/user/{user_id}")
async def get_user_engagement(
    user_id: str,
    user: User = Depends(get_current_user)
):
    """Get engagement metrics for a specific user"""
    if user.id != user_id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    profile = user_profiles.get(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    return {
        "user_id": user_id,
        "engagement_score": profile.calculate_engagement_score(),
        "total_sessions": profile.total_sessions,
        "total_messages": profile.total_messages,
        "success_rate": profile.successful_resolutions / max(profile.total_messages, 1),
        "frustration_rate": profile.frustration_indicators / max(profile.total_messages, 1),
        "first_seen": profile.first_seen.isoformat(),
        "last_seen": profile.last_seen.isoformat()
    }

@app.get("/api/engagement/session/{session_id}")
async def get_session_metrics(
    session_id: str,
    user: User = Depends(get_current_user)
):
    """Get metrics for a specific conversation session"""
    metrics = conversation_metrics.get(session_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "metrics": metrics.to_dict()
    }