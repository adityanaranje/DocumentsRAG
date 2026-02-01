import os
import json
import threading
import speech_recognition as sr
from flask import Flask, render_template, request, jsonify
from agents.graph import app as agent_app
from dotenv import load_dotenv
from ingestion.pipeline import IngestionPipeline
from rag.vector_store import VectorStoreManager

load_dotenv()

app = Flask(__name__)

# Global state for ingestion tracking
ingestion_status = {
    "status": "Idle",
    "progress": 0,
    "last_error": None
}

def ingest_worker(file_path, delete_source=None):
    """Worker thread for background ingestion using enhanced pipeline."""
    global ingestion_status
    try:
        ingestion_status["status"] = "Starting..."
        ingestion_status["progress"] = 0
        
        base_docs_dir = "docs"
        pipeline = IngestionPipeline(base_docs_dir)
        vector_manager = VectorStoreManager()
        
        if delete_source:
            ingestion_status["status"] = "Removing old version..."
            vector_manager.delete_documents_by_source(delete_source)
            ingestion_status["progress"] = 10
        
        ingestion_status["status"] = "Processing document..."
        ingestion_status["progress"] = 30
        
        # Use the new unified process_single_file method
        # Handles metadata extraction, section detection, and proper chunking
        chunks = pipeline.process_single_file(file_path)
        
        if chunks:
            ingestion_status["status"] = "Updating Vector Store..."
            ingestion_status["progress"] = 70
            vector_manager.update_vector_store(chunks)
            
            # Reload the retriever in the agent nodes to see new documents
            from agents.nodes import nodes
            nodes.reload_retriever()
            
            ingestion_status["status"] = "Completed Successfully!"
            ingestion_status["progress"] = 100
        else:
            ingestion_status["status"] = "Failed: No content extracted."
            ingestion_status["progress"] = 0
            
    except Exception as e:
        ingestion_status["status"] = "Failed"
        ingestion_status["last_error"] = str(e)
        ingestion_status["progress"] = 0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt")
    history = data.get("history", [])
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
        
    try:
        initial_state = {
            "input": prompt,
            "chat_history": history,
            "intent": "",
            "context": [],
            "answer": "",
            "metadata_filters": {}
        }
        
        result = agent_app.invoke(initial_state)
        return jsonify({
            "answer": result.get("answer", ""),
            "context": result.get("context", [])
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/api/audio-chat", methods=["POST"])
def audio_chat():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part"}), 400
    
    file = request.files['audio']
    history = json.loads(request.form.get("history", "[]"))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = "temp_voice_query.wav"
    file.save(temp_path)
    
    r = sr.Recognizer()
    try:
        with sr.AudioFile(temp_path) as source:
            audio_data = r.record(source)
        
        raw_text = r.recognize_google(audio_data)
        
        # Summarize/Refine the transcribed audio text
        from models.llm import LLMFactory
        from langchain_core.messages import SystemMessage, HumanMessage
        
        refiner_llm = LLMFactory.get_llm("small")
        refine_system = (
            "You are an assistant that cleans up and summarizes noisy speech-to-text transcriptions. "
            "Your goal is to extract the actual insurance-related question or request from the text.\n\n"
            "RULES:\n"
            "1. Remove filler words (um, ah, like, you know).\n"
            "2. Fix grammatical errors caused by transcription.\n"
            "3. If multiple things are mentioned, focus on the core request.\n"
            "4. Return ONLY the cleaned, professional question text."
        )
        
        refine_response = refiner_llm.invoke([
            SystemMessage(content=refine_system),
            HumanMessage(content=f"Transcription: {raw_text}")
        ])
        summarized_text = getattr(refine_response, 'content', str(refine_response)).strip()
        
        # Process with existing Agent using the summarized text
        initial_state = {
            "input": summarized_text,
            "chat_history": history,
            "intent": "",
            "context": [],
            "answer": "",
            "metadata_filters": {}
        }
        
        result = agent_app.invoke(initial_state)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({
            "transcription": raw_text,
            "summarized_question": summarized_text,
            "answer": result.get("answer", ""),
            "context": result.get("context", [])
        })
        
    except sr.UnknownValueError:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"error": "Could not understand audio"}), 400
    except sr.RequestError as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"error": f"Speech service error: {e}"}), 500
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

def update_doc_structure(provider_name, category_name):
    """Helper to persist new providers/categories to the config file."""
    try:
        config_path = os.path.join("configs", "doc_structure.json")
        if not os.path.exists(config_path):
            return
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Find or create provider
        provider = next((p for p in config["providers"] if p["name"] == provider_name), None)
        if not provider:
            # Insert at the beginning (before 'Other')
            provider = {"name": provider_name, "categories": []}
            config["providers"].insert(0, provider)
        
        # Add category if new
        if category_name not in provider["categories"]:
            provider["categories"].append(category_name)
            # Sort categories for cleanliness (except if it was General)
            if len(provider["categories"]) > 1:
                provider["categories"].sort()
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        pass

@app.route("/api/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    provider = request.form.get("provider")
    category = request.form.get("category")
    mode = request.form.get("mode", "New Upload") # "New Upload" or "Modify Existing"
    
    if file.filename == '' or not provider or not category:
        return jsonify({"error": "Missing metadata or file"}), 400

    # Persist new structure to JSON
    update_doc_structure(provider, category)

    base_dir = "docs"
    target_dir = os.path.join(base_dir, provider, category)
    os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, file.filename)
    file.save(file_path)
    
    delete_source = None
    if mode == "Modify Existing":
        file_to_modify = request.form.get("file_to_modify")
        if file_to_modify:
            delete_source = os.path.join(base_dir, provider, category, file_to_modify)
            if os.path.abspath(delete_source) != os.path.abspath(file_path):
                if os.path.exists(delete_source):
                    os.remove(delete_source)

    # Start background ingestion
    thread = threading.Thread(target=ingest_worker, args=(file_path, delete_source))
    thread.start()
    
    return jsonify({"message": "File uploaded, ingestion started.", "path": file_path})

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify(ingestion_status)

@app.route("/api/config", methods=["GET"])
def get_config():
    config_path = os.path.join("configs", "doc_structure.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"providers": []})

@app.route("/api/files", methods=["GET"])
def list_files():
    provider = request.args.get("provider")
    category = request.args.get("category")
    
    if not provider or not category:
        return jsonify({"files": []})
        
    base_dir = "docs"
    target_dir = os.path.join(base_dir, provider, category)
    if os.path.exists(target_dir):
        files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.pdf', '.docx'))]
        return jsonify({"files": files})
    return jsonify({"files": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
