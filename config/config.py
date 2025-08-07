"""
Configuration settings for the LangChain RAG Application
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    temperature: float = 0.7
    max_new_tokens: int = 512
    load_in_8bit: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    name: str
    device: str = "cpu"

class Config:
    """Application configuration with model presets and flexible settings"""
    
    # ===== MODEL PRESETS =====
    # Easy-to-use model configurations
    MODEL_PRESETS = {
        # Fast, lightweight models
        "fast": {
            "llm": ModelConfig(
                name="microsoft/phi-2",
                temperature=0.7,
                max_new_tokens=512,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        # Balanced performance (default) - Updated to modern model
        "balanced": {
            "llm": ModelConfig(
                name="microsoft/phi-2",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        # High quality, modern models
        "quality": {
            "llm": ModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.3",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-mpnet-base-v2"
            )
        },
        
        # Maximum quality (requires more RAM)
        "max_quality": {
            "llm": ModelConfig(
                name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-mpnet-base-v2"
            )
        },
        
        # Technical/Coding focused
        "technical": {
            "llm": ModelConfig(
                name="codellama/CodeLlama-7b-Instruct-hf",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        # Legacy models (for backward compatibility)
        "gpt2": {
            "llm": ModelConfig(
                name="gpt2",
                temperature=0.7,
                max_new_tokens=512,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        "llama2": {
            "llm": ModelConfig(
                name="meta-llama/Llama-2-7b-chat-hf",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        # Additional modern options
        "mistral": {
            "llm": ModelConfig(
                name="mistralai/Mistral-7B-Instruct-v0.3",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-mpnet-base-v2"
            )
        },
        
        "mixtral": {
            "llm": ModelConfig(
                name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-mpnet-base-v2"
            )
        },
        
        "phi": {
            "llm": ModelConfig(
                name="microsoft/phi-2",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        },
        
        "gemma": {
            "llm": ModelConfig(
                name="google/gemma-7b-it",
                temperature=0.7,
                max_new_tokens=1024,
                load_in_8bit=True
            ),
            "embedding": EmbeddingConfig(
                name="sentence-transformers/all-MiniLM-L6-v2"
            )
        }
    }
    
    # ===== CURRENT CONFIGURATION =====
    # Get preset from environment or use default
    CURRENT_PRESET = os.getenv("MODEL_PRESET", "balanced")
    
    # Override with custom model if specified
    CUSTOM_LLM_MODEL = os.getenv("CUSTOM_LLM_MODEL", None)
    CUSTOM_EMBEDDING_MODEL = os.getenv("CUSTOM_EMBEDDING_MODEL", None)
    
    # ===== ADVANCED SETTINGS =====
    # These can override preset values
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "true").lower() == "true"
    DEVICE_MAP = os.getenv("DEVICE_MAP", "auto")
    TORCH_DTYPE = os.getenv("TORCH_DTYPE", "auto")
    
    # ===== DEVICE SETTINGS =====
    # Force device selection: "auto", "cpu", "cuda", "mps" (for Apple Silicon)
    FORCE_DEVICE = os.getenv("FORCE_DEVICE", "auto")
    # Enable GPU acceleration for embeddings
    EMBEDDING_USE_GPU = os.getenv("EMBEDDING_USE_GPU", "auto").lower() == "true"
    
    # ===== DOCUMENT PROCESSING SETTINGS =====
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # ===== ENHANCED CHUNKING SETTINGS =====
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "adaptive")  # Options: recursive, semantic, adaptive
    MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))
    MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    
    # ===== RETRIEVAL SETTINGS =====
    RETRIEVAL_METHOD = os.getenv("RETRIEVAL_METHOD", "mmr")  # Options: similarity, mmr, hybrid
    MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # MMR diversity parameter (0-1)
    FETCH_K = int(os.getenv("FETCH_K", "20"))  # Number of docs to fetch before MMR selection
    HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))  # Weight for semantic vs keyword search
    
    # ===== VECTOR STORE SETTINGS =====
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
    NUM_RETRIEVED_DOCS = int(os.getenv("NUM_RETRIEVED_DOCS", "4"))
    
    # ===== FILE STORAGE SETTINGS =====
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents")
    SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]
    
    # ===== SYSTEM SETTINGS =====
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    CACHE_DIR = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    @classmethod
    def get_current_llm_config(cls) -> ModelConfig:
        """Get current LLM configuration"""
        # Get optimal device configuration
        device_config = cls.get_optimal_device_config()
        
        if cls.CUSTOM_LLM_MODEL:
            # Use custom model with default settings
            return ModelConfig(
                name=cls.CUSTOM_LLM_MODEL,
                temperature=cls.TEMPERATURE,
                max_new_tokens=cls.MAX_NEW_TOKENS,
                load_in_8bit=cls.LOAD_IN_8BIT,
                device_map=device_config["llm_device_map"],
                torch_dtype=cls.TORCH_DTYPE
            )
        
        # Use preset configuration
        preset = cls.MODEL_PRESETS.get(cls.CURRENT_PRESET, cls.MODEL_PRESETS["balanced"])
        llm_config = preset["llm"]
        
        # Override with environment variables if set
        if cls.TEMPERATURE != 0.7:
            llm_config.temperature = cls.TEMPERATURE
        if cls.MAX_NEW_TOKENS != 512:
            llm_config.max_new_tokens = cls.MAX_NEW_TOKENS
        if cls.LOAD_IN_8BIT != True:
            llm_config.load_in_8bit = cls.LOAD_IN_8BIT
        
        # Apply optimal device configuration
        llm_config.device_map = device_config["llm_device_map"]
        if cls.TORCH_DTYPE != "auto":
            llm_config.torch_dtype = cls.TORCH_DTYPE
            
        return llm_config
    
    @classmethod
    def get_current_embedding_config(cls) -> EmbeddingConfig:
        """Get current embedding configuration"""
        # Get optimal device configuration
        device_config = cls.get_optimal_device_config()
        
        if cls.CUSTOM_EMBEDDING_MODEL:
            return EmbeddingConfig(
                name=cls.CUSTOM_EMBEDDING_MODEL,
                device=device_config["embedding_device"]
            )
        
        # Use preset configuration
        preset = cls.MODEL_PRESETS.get(cls.CURRENT_PRESET, cls.MODEL_PRESETS["balanced"])
        embedding_config = preset["embedding"]
        
        # Apply optimal device configuration
        embedding_config.device = device_config["embedding_device"]
        
        return embedding_config
    
    @classmethod
    def get_all_llm_configs(cls) -> List[Dict[str, Any]]:
        """Get all available LLM configurations for fallback"""
        device_config = cls.get_optimal_device_config()
        
        configs = []
        
        # Add current configuration first
        current_config = {
            "name": cls.get_current_llm_config().name,
            "temperature": cls.TEMPERATURE,
            "load_in_8bit": cls.LOAD_IN_8BIT,
            "device_map": device_config["llm_device_map"],
            "torch_dtype": cls.TORCH_DTYPE
        }
        configs.append(current_config)
        
        # Add fallback configurations
        fallback_presets = ["fast", "balanced", "phi"]  # Only open-access models
        
        for preset_name in fallback_presets:
            if preset_name != cls.CURRENT_PRESET:
                preset = cls.MODEL_PRESETS.get(preset_name)
                if preset:
                    config = {
                        "name": preset["llm"].name,
                        "temperature": preset["llm"].temperature,
                        "load_in_8bit": preset["llm"].load_in_8bit,
                        "device_map": device_config["llm_device_map"],
                        "torch_dtype": cls.TORCH_DTYPE
                    }
                    configs.append(config)
        
        return configs
    
    @classmethod
    def get_available_presets(cls) -> Dict[str, Dict[str, Any]]:
        """Get all available presets with descriptions"""
        return {
            "fast": {
                "description": "Fast and lightweight - Microsoft Phi-2 (excellent for quick testing)",
                "llm_model": "microsoft/phi-2",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~3GB",
                "speed": "Fast"
            },
            "balanced": {
                "description": "Balanced performance - Microsoft Phi-2 (recommended default)",
                "llm_model": "microsoft/phi-2", 
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~3GB",
                "speed": "Fast"
            },
            "quality": {
                "description": "High quality - Mistral 7B Instruct (excellent performance)",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.3",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "memory_usage": "~8GB",
                "speed": "Medium"
            },
            "max_quality": {
                "description": "Maximum quality - Mixtral 8x7B (state-of-the-art performance)",
                "llm_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "memory_usage": "~16GB",
                "speed": "Slow"
            },
            "technical": {
                "description": "Technical/Coding focused - Code Llama 7B (excellent for code/docs)",
                "llm_model": "codellama/CodeLlama-7b-Instruct-hf",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~8GB",
                "speed": "Medium"
            },
            "mistral": {
                "description": "Mistral 7B Instruct - excellent all-around performance",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.3",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "memory_usage": "~8GB",
                "speed": "Medium"
            },
            "mixtral": {
                "description": "Mixtral 8x7B - state-of-the-art performance (requires more RAM)",
                "llm_model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "memory_usage": "~16GB",
                "speed": "Slow"
            },
            "phi": {
                "description": "Microsoft Phi-2 - excellent performance for its size",
                "llm_model": "microsoft/phi-2",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~3GB",
                "speed": "Fast"
            },
            "gemma": {
                "description": "Google Gemma 7B - good performance, open source",
                "llm_model": "google/gemma-7b-it",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~8GB",
                "speed": "Medium"
            },
            "gpt2": {
                "description": "Legacy GPT-2 model (for backward compatibility)",
                "llm_model": "gpt2",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~500MB",
                "speed": "Fast"
            },
            "llama2": {
                "description": "Llama 2 7B model (requires HuggingFace access)",
                "llm_model": "meta-llama/Llama-2-7b-chat-hf",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "memory_usage": "~8GB",
                "speed": "Medium"
            }
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        # Check if preset exists
        if cls.CURRENT_PRESET not in cls.MODEL_PRESETS and not cls.CUSTOM_LLM_MODEL:
            raise ValueError(f"Invalid preset: {cls.CURRENT_PRESET}. Available: {list(cls.MODEL_PRESETS.keys())}")
        
        # Check if custom models are specified
        if cls.CUSTOM_LLM_MODEL and not cls.CUSTOM_LLM_MODEL.strip():
            raise ValueError("CUSTOM_LLM_MODEL cannot be empty")
        
        if cls.CUSTOM_EMBEDDING_MODEL and not cls.CUSTOM_EMBEDDING_MODEL.strip():
            raise ValueError("CUSTOM_EMBEDDING_MODEL cannot be empty")
        
        return True
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all current settings as a dictionary"""
        llm_config = cls.get_current_llm_config()
        embedding_config = cls.get_current_embedding_config()
        
        return {
            "preset": cls.CURRENT_PRESET,
            "llm_model": llm_config.name,
            "llm_temperature": llm_config.temperature,
            "llm_max_new_tokens": llm_config.max_new_tokens,
            "llm_load_in_8bit": llm_config.load_in_8bit,
            "llm_device_map": llm_config.device_map,
            "llm_torch_dtype": llm_config.torch_dtype,
            "embedding_model": embedding_config.name,
            "embedding_device": embedding_config.device,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "vector_db_path": cls.VECTOR_DB_PATH,
            "num_retrieved_docs": cls.NUM_RETRIEVED_DOCS,
            "documents_path": cls.DOCUMENTS_PATH,
            "supported_file_types": cls.SUPPORTED_FILE_TYPES,
            "log_level": cls.LOG_LEVEL,
            "cache_dir": cls.CACHE_DIR
        }
    
    @classmethod
    def detect_available_devices(cls) -> Dict[str, Any]:
        """Detect available devices and their capabilities"""
        devices = {
            "cpu": True,
            "cuda": False,
            "mps": False,
            "recommended_device": "cpu"
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                devices["cuda"] = True
                devices["recommended_device"] = "cuda"
                devices["cuda_count"] = torch.cuda.device_count()
                devices["cuda_name"] = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                devices["mps"] = True
                devices["recommended_device"] = "mps"
        except ImportError:
            pass
        
        return devices
    
    @classmethod
    def get_optimal_device_config(cls) -> Dict[str, str]:
        """Get optimal device configuration based on available hardware"""
        devices = cls.detect_available_devices()
        
        # If user forced a specific device, use it
        if cls.FORCE_DEVICE != "auto":
            return {
                "llm_device_map": cls.FORCE_DEVICE if cls.FORCE_DEVICE != "cpu" else "cpu",
                "embedding_device": cls.FORCE_DEVICE
            }
        
        # Auto-detect optimal configuration
        if devices["cuda"]:
            return {
                "llm_device_map": "auto",  # Let transformers handle GPU allocation
                "embedding_device": "cuda"
            }
        elif devices["mps"]:
            return {
                "llm_device_map": "auto",
                "embedding_device": "mps"
            }
        else:
            return {
                "llm_device_map": "cpu",
                "embedding_device": "cpu"
            }
    
    @classmethod
    def print_current_config(cls):
        """Print current configuration for debugging"""
        print(f"\n🔧 Current Configuration:")
        print(f"   Preset: {cls.CURRENT_PRESET}")
        
        # Device information
        devices = cls.detect_available_devices()
        print(f"\n🖥️  Device Information:")
        print(f"   CPU: Available")
        print(f"   CUDA: {'Available' if devices['cuda'] else 'Not available'}")
        if devices['cuda']:
            print(f"   CUDA Devices: {devices['cuda_count']} - {devices['cuda_name']}")
        print(f"   MPS (Apple Silicon): {'Available' if devices['mps'] else 'Not available'}")
        print(f"   Recommended Device: {devices['recommended_device']}")
        
        # Current device configuration
        device_config = cls.get_optimal_device_config()
        print(f"\n⚙️  Device Configuration:")
        print(f"   LLM Device Map: {device_config['llm_device_map']}")
        print(f"   Embedding Device: {device_config['embedding_device']}")
        print(f"   Force Device: {cls.FORCE_DEVICE}")
        print(f"   Embedding GPU: {cls.EMBEDDING_USE_GPU}")
        
        llm_config = cls.get_current_llm_config()
        print(f"\n🤖 Model Configuration:")
        print(f"   LLM Model: {llm_config.name}")
        print(f"   LLM Temperature: {llm_config.temperature}")
        print(f"   LLM Max Tokens: {llm_config.max_new_tokens}")
        print(f"   LLM 8-bit: {llm_config.load_in_8bit}")
        
        embedding_config = cls.get_current_embedding_config()
        print(f"   Embedding Model: {embedding_config.name}")
        print(f"   Embedding Device: {embedding_config.device}")
        
        print(f"\n📄 Processing Configuration:")
        print(f"   Chunk Size: {cls.CHUNK_SIZE}")
        print(f"   Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"   Retrieved Docs: {cls.NUM_RETRIEVED_DOCS}")
        print() 