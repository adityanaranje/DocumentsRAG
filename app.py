import os
import json
import threading
import uuid
import time
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from agents.graph import app as agent_app
from dotenv import load_dotenv
from ingestion.pipeline import IngestionPipeline
from rag.vector_store import VectorStoreManager

# Production imports
from config import config
from utils.logger import setup_logger, set_request_context, clear_request_context
from utils.validators import InputValidator, ValidationError
from utils.metrics import metrics
from utils.request_logger import request_logger
from utils.cache import cache_manager
from api.monitoring import monitoring_bp

load_dotenv()

# Setup logging
logger = setup_logger(__name__)

app = Flask(__name__)

# Register monitoring blueprint
app.register_blueprint(monitoring_bp)

# Configure CORS
if config.ENABLE_CORS:
    CORS(app, origins=config.CORS_ORIGINS)
    logger.info(f"CORS enabled for origins: {config.CORS_ORIGINS}")

# Global state for ingestion tracking
ingestion_status = {
    "status": "Idle",
    "progress": 0,
    "last_error": None
}


@app.before_request
def before_request():
    """Set up request context and tracking."""
    # Generate request ID
    request_id = str(uuid.uuid4())[:8]
    request.request_id = request_id
    request.start_time = time.time()
    
    # Set request context for logging
    user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    set_request_context(request_id, user_ip)
    
    # Track active requests
    metrics.increment_active_requests()
    
    # Log request
    logger.info(f"Request started: {request.method} {request.path}")


@app.after_request
def after_request(response):
    """Clean up request context and record metrics."""
    if hasattr(request, 'start_time'):
        latency_ms = (time.time() - request.start_time) * 1000
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.path} "
            f"[{response.status_code}] {latency_ms:.2f}ms"
        )
        
        # Record metrics
        metrics.record_request(
            latency_ms=latency_ms,
            error=(response.status_code >= 400)
        )
    
    # Decrement active requests
    metrics.decrement_active_requests()
    
    # Clear request context
    clear_request_context()
    
    return response


@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Unhandled error: {str(error)}", exc_info=True)
    
    # Don't expose internal errors in production
    if config.DEBUG:
        error_msg = str(error)
    else:
        error_msg = "An internal error occurred. Please try again later."
    
    return jsonify({
        "error": error_msg,
        "request_id": getattr(request, 'request_id', 'unknown')
    }), 500


def ingest_worker(file_path, delete_source=None):
    """Worker thread for background ingestion using enhanced pipeline."""
    global ingestion_status
    try:
        logger.info(f"Starting ingestion for: {file_path}")
        ingestion_status["status"] = "Starting..."
        ingestion_status["progress"] = 0
        
        base_docs_dir = config.DOCS_DIR
        pipeline = IngestionPipeline(base_docs_dir)
        vector_manager = VectorStoreManager()
        
        if delete_source:
            logger.info(f"Removing old version: {delete_source}")
            ingestion_status["status"] = "Removing old version..."
            vector_manager.delete_documents_by_source(delete_source)
            ingestion_status["progress"] = 10
        
        ingestion_status["status"] = "Processing document..."
        ingestion_status["progress"] = 30
        
        chunks = pipeline.process_single_file(file_path)
        
        if chunks:
            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
            ingestion_status["status"] = "Updating Vector Store..."
            ingestion_status["progress"] = 70
            vector_manager.update_vector_store(chunks)
            
            # Invalidate caches
            cache_manager.invalidate_all()
            logger.info("Caches invalidated after ingestion")
            
            # Reload the retriever in the agent nodes
            from agents.nodes import nodes
            nodes.reload_retriever()
            
            ingestion_status["status"] = "Completed Successfully!"
            ingestion_status["progress"] = 100
            logger.info(f"Ingestion completed successfully for: {file_path}")
        else:
            logger.warning(f"No content extracted from: {file_path}")
            ingestion_status["status"] = "Failed: No content extracted."
            ingestion_status["progress"] = 0
            
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        ingestion_status["status"] = "Failed"
        ingestion_status["last_error"] = str(e)
        ingestion_status["progress"] = 0


@app.route("/")
def index():
    """Serve main page."""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint with full error handling and logging."""
    start_time = time.time()
    request_id = getattr(request, 'request_id', 'unknown')
    
    try:
        data = request.json
        if not data:
            raise ValidationError("Request body must be JSON")
        
        prompt = data.get("prompt")
        history = data.get("history", [])
        extracted_entities = data.get("extracted_entities", {})
        
        # Validate input
        if not prompt:
            raise ValidationError("Prompt is required")
        
        InputValidator.validate_query_input(prompt)
        
        logger.info(f"Chat request: {prompt[:100]}...")
        
        # Process with agent
        initial_state = {
            "input": prompt,
            "chat_history": history,
            "intent": "",
            "context": [],
            "answer": "",
            "metadata_filters": {},
            "extracted_entities": extracted_entities
        }
        
        result = agent_app.invoke(initial_state)
        
        # Extract results
        answer = result.get("answer", "")
        context = result.get("context", [])
        intent = result.get("intent", "unknown")
        entities = result.get("extracted_entities", {})
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Log request to database
        request_logger.log_request(
            request_id=request_id,
            query=prompt,
            intent=intent,
            extracted_entities=entities,
            retrieval_count=len(context),
            latency_ms=latency_ms,
            status="success",
            context_sources=[c[:100] for c in context[:5]],  # First 5 sources
            user_ip=request.headers.get('X-Forwarded-For', request.remote_addr)
        )
        
        # Record intent in metrics
        metrics.record_request(latency_ms=latency_ms, intent=intent, error=False)
        
        logger.info(f"Chat completed successfully. Intent: {intent}, Latency: {latency_ms:.2f}ms")
        
        return jsonify({
            "answer": answer,
            "context": context,
            "extracted_entities": entities,
            "intent": intent,
            "request_id": request_id
        })
        
    except ValidationError as e:
        logger.warning(f"Validation error: {str(e)}")
        latency_ms = (time.time() - start_time) * 1000
        
        request_logger.log_request(
            request_id=request_id,
            query=data.get("prompt", "")[:500] if data else "",
            latency_ms=latency_ms,
            status="validation_error",
            error_message=str(e),
            user_ip=request.headers.get('X-Forwarded-For', request.remote_addr)
        )
        
        return jsonify({
            "error": str(e),
            "request_id": request_id
        }), 400
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        latency_ms = (time.time() - start_time) * 1000
        
        request_logger.log_request(
            request_id=request_id,
            query=data.get("prompt", "")[:500] if data else "",
            latency_ms=latency_ms,
            status="error",
            error_message=str(e)[:500],
            user_ip=request.headers.get('X-Forwarded-For', request.remote_addr)
        )
        
        error_msg = str(e) if config.DEBUG else "An error occurred processing your request"
        return jsonify({
            "error": error_msg,
            "request_id": request_id,
            "status": "error"
        }), 500


@app.route("/api/audio-chat", methods=["POST"])
def audio_chat():
    """Audio chat endpoint with validation."""
    start_time = time.time()
    request_id = getattr(request, 'request_id', 'unknown')
    temp_path = None
    
    try:
        if 'audio' not in request.files:
            raise ValidationError("No audio file provided")
        
        file = request.files['audio']
        history = json.loads(request.form.get("history", "[]"))
        extracted_entities = json.loads(request.form.get("extracted_entities", "{}"))
        
        if file.filename == '':
            raise ValidationError("No file selected")
        
        # Save temporarily
        temp_path = f"temp_voice_{request_id}.wav"
        file.save(temp_path)
        
        logger.info(f"Processing audio file: {file.filename}")
        
        # Transcribe
        r = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = r.record(source)
        
        raw_text = r.recognize_google(audio_data)
        logger.info(f"Transcribed: {raw_text}")
        
        # Summarize/refine transcription
        from models.llm import LLMFactory
        from langchain_core.messages import SystemMessage, HumanMessage
        
        refiner_llm = LLMFactory.get_llm("small")
        refine_system = (
            "You are an assistant that cleans up and summarizes noisy speech-to-text transcriptions. "
            "Your goal is to extract the actual insurance-related question or request from the text.\\n\\n"
            "RULES:\\n"
            "1. Remove filler words (um, ah, like, you know).\\n"
            "2. Fix grammatical errors caused by transcription.\\n"
            "3. If multiple things are mentioned, focus on the core request.\\n"
            "4. Return ONLY the cleaned, professional question text."
        )
        
        refine_response = refiner_llm.invoke([
            SystemMessage(content=refine_system),
            HumanMessage(content=f"Transcription: {raw_text}")
        ])
        summarized_text = getattr(refine_response, 'content', str(refine_response)).strip()
        
        logger.info(f"Refined: {summarized_text}")
        
        # Process with agent (similar to chat endpoint)
        initial_state = {
            "input": summarized_text,
            "chat_history": history,
            "intent": "",
            "context": [],
            "answer": "",
            "metadata_filters": {},
            "extracted_entities": extracted_entities
        }
        
        result = agent_app.invoke(initial_state)
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.info(f"Audio chat completed. Latency: {latency_ms:.2f}ms")
        
        return jsonify({
            "transcription": raw_text,
            "summarized_question": summarized_text,
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "extracted_entities": result.get("extracted_entities", {}),
            "request_id": request_id
        })
        
    except sr.UnknownValueError:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.warning("Could not understand audio")
        return jsonify({
            "error": "Could not understand audio",
            "request_id": request_id
        }), 400
        
    except sr.RequestError as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Speech service error: {e}")
        return jsonify({
            "error": f"Speech service error: {e}",
            "request_id": request_id
        }), 500
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Audio chat error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e) if config.DEBUG else "Error processing audio",
            "request_id": request_id
        }), 500


def update_doc_structure(provider_name, category_name):
    """Helper to persist new providers/categories to the config file."""
    try:
        config_path = os.path.join("configs", "doc_structure.json")
        if not os.path.exists(config_path):
            return
        
        with open(config_path, "r") as f:
            doc_config = json.load(f)
        
        # Find or create provider
        provider = next((p for p in doc_config["providers"] if p["name"] == provider_name), None)
        if not provider:
            provider = {"name": provider_name, "categories": []}
            doc_config["providers"].insert(0, provider)
        
        # Add category if new
        if category_name not in provider["categories"]:
            provider["categories"].append(category_name)
            if len(provider["categories"]) > 1:
                provider["categories"].sort()
        
        with open(config_path, "w") as f:
            json.dump(doc_config, f, indent=4)
            
    except Exception as e:
        logger.warning(f"Failed to update doc structure: {e}")


@app.route("/api/upload", methods=["POST"])
def upload():
    """File upload endpoint with validation."""
    try:
        if 'file' not in request.files:
            raise ValidationError("No file provided")
        
        file = request.files['file']
        provider = request.form.get("provider")
        category = request.form.get("category")
        mode = request.form.get("mode", "New Upload")
        
        if file.filename == '' or not provider or not category:
            raise ValidationError("Missing required fields: file, provider, or category")
        
        # Validate file
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        InputValidator.validate_file_upload(file.filename, file_size)
        
        # Sanitize filename
        safe_filename = InputValidator.sanitize_filename(file.filename)
        logger.info(f"Uploading file: {safe_filename} ({file_size} bytes)")
        
        # Update doc structure
        update_doc_structure(provider, category)
        
        # Save file
        base_dir = config.DOCS_DIR
        target_dir = os.path.join(base_dir, provider, category)
        os.makedirs(target_dir, exist_ok=True)
        
        file_path = os.path.join(target_dir, safe_filename)
        file.save(file_path)
        
        logger.info(f"File saved to: {file_path}")
        
        # Handle file modification
        delete_source = None
        if mode == "Modify Existing":
            file_to_modify = request.form.get("file_to_modify")
            if file_to_modify:
                delete_source = os.path.join(base_dir, provider, category, file_to_modify)
                if os.path.abspath(delete_source) != os.path.abspath(file_path):
                    if os.path.exists(delete_source):
                        os.remove(delete_source)
                        logger.info(f"Removed old file: {delete_source}")
        
        # Start background ingestion
        thread = threading.Thread(target=ingest_worker, args=(file_path, delete_source))
        thread.start()
        
        return jsonify({
            "message": "File uploaded successfully, ingestion started.",
            "filename": safe_filename,
            "path": file_path
        })
        
    except ValidationError as e:
        logger.warning(f"Upload validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e) if config.DEBUG else "Upload failed"
        }), 500


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get ingestion status."""
    return jsonify(ingestion_status)


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get document structure configuration."""
    config_path = os.path.join("configs", "doc_structure.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"providers": []})


@app.route("/api/files", methods=["GET"])
def list_files():
    """List files in a provider/category directory."""
    provider = request.args.get("provider")
    category = request.args.get("category")
    
    if not provider or not category:
        return jsonify({"files": []})
    
    base_dir = config.DOCS_DIR
    target_dir = os.path.join(base_dir, provider, category)
    
    if os.path.exists(target_dir):
        files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.pdf', '.docx'))]
        return jsonify({"files": files})
    
    return jsonify({"files": []})


if __name__ == "__main__":
    # Log configuration on startup
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION}")
    logger.info(f"Environment: {config.ENVIRONMENT.value}")
    logger.info(f"Configuration: {json.dumps(config.get_summary(), indent=2)}")
    
    # Validate configuration
    try:
        config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
    
    # Start application
    port = config.PORT
    host = config.HOST
    debug = config.DEBUG
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
