"""
RAG Module - Retrieval Augmented Generation con ChromaDB y Ollama.

Proporciona búsqueda semántica sobre documentos locales usando:
- ChromaDB: Base de datos vectorial local
- Ollama embeddings: nomic-embed-text (ligero y eficiente)

Uso:
    from rag_module import RAGSystem
    
    rag = RAGSystem()
    rag.add_documents(["texto1", "texto2"], ["id1", "id2"])
    results = rag.search("mi consulta")
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
import httpx
import numpy as np

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
RAG_DATA_DIR = Path("rag_data")
INDEX_FILE = RAG_DATA_DIR / ".index.json"
DOCUMENTS_DIR = Path("documents")

# Ollama embedding configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"  # ~275MB, muy eficiente
# Límite seguro de caracteres por texto para el modelo de embeddings.
# nomic-embed-text tiene 8192 tokens de contexto (~4 chars/token → 32K).
# Usamos 28 000 chars como margen de seguridad.
MAX_EMBED_CHARS = 28_000


class OllamaEmbeddingFunction:
    """Función de embeddings usando Ollama, compatible con ChromaDB >= 1.5."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self.url = f"{OLLAMA_BASE_URL}/api/embed"
    
    def name(self) -> str:
        """Return embedding function name (required by ChromaDB)."""
        return f"ollama_{self.model_name}"
    
    def _get_embeddings(self, texts: List[str], allow_fallback: bool = True) -> List[List[float]]:
        """Genera embeddings para una lista de strings planos.
        
        Args:
            texts: Textos a embeddear
            allow_fallback: Si True, retorna vectores nulos en caso de error.
                            Si False, lanza excepción (usado en ingesta).
        """
        # Truncar textos que excedan el contexto del modelo de embeddings
        safe_texts = []
        for t in texts:
            if len(t) > MAX_EMBED_CHARS:
                print(f"⚠️ Embedding text truncated: {len(t)} → {MAX_EMBED_CHARS} chars")
                safe_texts.append(t[:MAX_EMBED_CHARS])
            else:
                safe_texts.append(t)
        try:
            response = httpx.post(
                self.url,
                json={"model": self.model_name, "input": safe_texts},
                timeout=120.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["embeddings"]
            else:
                msg = f"Error getting embeddings: {response.status_code} - {response.text[:200]}"
                print(msg)
                if not allow_fallback:
                    raise RuntimeError(msg)
                return [[0.0] * 768 for _ in safe_texts]
        except (RuntimeError,):
            raise
        except Exception as e:
            msg = f"Embedding error: {e}"
            print(msg)
            if not allow_fallback:
                raise RuntimeError(msg) from e
            return [[0.0] * 768 for _ in safe_texts]

    def __call__(self, input) -> List[np.ndarray]:
        """
        Genera embeddings. ChromaDB 1.5 pasa input como List[List[str]] (lista de listas).
        Versiones anteriores pasan List[str].
        Retorna List[np.ndarray] como requiere ChromaDB 1.5.
        
        Propagates errors (allow_fallback=False) so failed embeddings don't
        produce zero-vector documents in ChromaDB.
        """
        # Flatten: ChromaDB 1.5 sends [['text1'], ['text2']] instead of ['text1', 'text2']
        flat_texts = []
        if input and isinstance(input[0], list):
            for item in input:
                flat_texts.append(item[0] if item else "")
        else:
            flat_texts = list(input)
        
        raw = self._get_embeddings(flat_texts, allow_fallback=False)
        return [np.array(e, dtype=np.float32) for e in raw]

    def embed_query(self, input = "", query: str = "", **kwargs) -> List[np.ndarray]:
        """Genera embedding para una query (requerido por ChromaDB >= 1.5).
        Returns List[np.ndarray] (Embeddings type) — same as __call__."""
        # ChromaDB 1.5 passes input as a list: embed_query(input=['text'])
        if isinstance(input, list):
            texts = [t if isinstance(t, str) else str(t) for t in input]
        else:
            texts = [input or query]
        raw = self._get_embeddings(texts)
        return [np.array(e, dtype=np.float32) for e in raw]

    def embed_documents(self, input = None, documents: List[str] = None, **kwargs) -> List[np.ndarray]:
        """Genera embeddings para documentos (requerido por ChromaDB).
        Propagates errors so failed embeddings don't produce zero-vectors."""
        texts = input or documents or []
        # Flatten if ChromaDB passes nested lists: [['text1'], ['text2']]
        if texts and isinstance(texts[0], list):
            texts = [t[0] if t else "" for t in texts]
        raw = self._get_embeddings(texts, allow_fallback=False)
        return [np.array(e, dtype=np.float32) for e in raw]


class RAGSystem:
    """Sistema RAG con ChromaDB y Ollama embeddings."""
    
    def __init__(self, collection_name: str = "knowledge_base"):
        """
        Inicializa el sistema RAG.
        
        Args:
            collection_name: Nombre de la colección en ChromaDB
        """
        RAG_DATA_DIR.mkdir(exist_ok=True)
        
        # Cliente persistente de ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(RAG_DATA_DIR),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Función de embeddings con Ollama
        self.embedding_fn = OllamaEmbeddingFunction()
        
        # Obtener o crear colección
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collection_name = collection_name
    
    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadatas: Optional[List[Dict]] = None
    ) -> int:
        """
        Agrega documentos a la base de conocimiento.
        Processes chunks individually so one embedding failure doesn't
        discard the entire batch.
        
        Args:
            documents: Lista de textos a indexar
            ids: IDs únicos para cada documento
            metadatas: Metadata opcional para cada documento
        
        Returns:
            Número de documentos agregados exitosamente
        """
        if not documents:
            return 0
        
        # Asegurar que tenemos metadata para cada documento
        if metadatas is None:
            metadatas = [{"indexed_at": datetime.now().isoformat()} for _ in documents]
        
        added = 0
        failed = 0
        # Process in small batches of 5 to balance speed vs. resilience
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            try:
                self.collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_meta
                )
                added += len(batch_docs)
            except Exception as e:
                # Batch failed — fall back to individual inserts
                print(f"⚠️ Batch embedding failed, trying individually: {e}")
                for j in range(len(batch_docs)):
                    try:
                        self.collection.add(
                            documents=[batch_docs[j]],
                            ids=[batch_ids[j]],
                            metadatas=[batch_meta[j]]
                        )
                        added += 1
                    except Exception as e2:
                        failed += 1
                        src = batch_meta[j].get('source', batch_ids[j]) if batch_meta else batch_ids[j]
                        print(f"❌ Skipped chunk {batch_ids[j]} ({src}): {e2}")
        
        if failed:
            print(f"⚠️ add_documents: {added} added, {failed} failed")
        return added
    
    def search(
        self,
        query: str,
        n_results: int = 3,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Busca documentos relevantes.
        
        Args:
            query: Consulta de búsqueda
            n_results: Número de resultados a retornar
            where: Filtro opcional de metadata
        
        Returns:
            Lista de resultados con documento, metadata y distancia
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Formatear resultados
            formatted = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0,
                        "relevance": 1 - (results["distances"][0][i] if results["distances"] else 0)
                    })
            
            return formatted
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def delete_by_source(self, source: str) -> int:
        """
        Elimina todos los documentos de una fuente.
        
        Args:
            source: Nombre del archivo fuente
        
        Returns:
            Número de documentos eliminados
        """
        try:
            # Buscar IDs que coincidan con la fuente
            results = self.collection.get(
                where={"source": source},
                include=[]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                return len(results["ids"])
            return 0
        except Exception as e:
            print(f"Delete error: {e}")
            return 0
    
    def delete_by_ids(self, ids: List[str]) -> int:
        """Elimina documentos por IDs."""
        try:
            self.collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print(f"Delete error: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas de la base de conocimiento."""
        return {
            "collection_name": self.collection_name,
            "total_documents": self.collection.count(),
            "embedding_model": EMBEDDING_MODEL
        }
    
    def close(self):
        """Release ChromaDB client resources to allow clean re-initialization."""
        try:
            if hasattr(self, 'client') and self.client:
                # Release internal resources so a new PersistentClient
                # on the same path reads the latest data from disk.
                if hasattr(self.client, '_identifier_to_system'):
                    self.client._identifier_to_system.clear()
                del self.collection
                del self.client
                self.client = None
                self.collection = None
        except Exception as e:
            print(f"Close warning: {e}")

    def clear(self):
        """Elimina todos los documentos de la colección."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Clear error: {e}")


# --------------------------------------------------------------------
# Index Management
# --------------------------------------------------------------------

def load_index() -> Dict:
    """Carga el índice de archivos procesados."""
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())
    return {"files": {}, "last_sync": None}


def save_index(index: Dict):
    """Guarda el índice."""
    INDEX_FILE.parent.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index, indent=2, default=str))


def get_file_hash(file_path: Path) -> str:
    """Calcula hash MD5 de un archivo."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


# --------------------------------------------------------------------
# RAG Prompt Builder
# --------------------------------------------------------------------

def build_rag_prompt(
    query: str,
    context_docs: List[Dict],
    system_instruction: str = None
) -> str:
    """
    Construye el prompt con contexto RAG.
    
    Args:
        query: Pregunta del usuario
        context_docs: Documentos de contexto del RAG search
        system_instruction: Instrucción de sistema opcional
    
    Returns:
        Prompt formateado con contexto
    """
    if not context_docs:
        return query
    
    # Formatear contexto
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("metadata", {}).get("source", "Documento")
        section = doc.get("metadata", {}).get("section", "")
        header = f"[{source}]" + (f" - {section}" if section else "")
        context_parts.append(f"### {header}\n{doc['content']}")
    
    context_text = "\n\n".join(context_parts)
    
    # Construir prompt
    base_instruction = system_instruction or (
        "Usa la siguiente información de contexto para responder la pregunta del usuario. "
        "Si la respuesta no está en el contexto, indícalo claramente. "
        "Cita la fuente cuando sea relevante."
    )
    
    return f"""{base_instruction}

## Contexto:
{context_text}

## Pregunta:
{query}

## Respuesta:"""


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

async def check_embedding_model() -> bool:
    """Verifica si el modelo de embeddings está disponible en Ollama."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                # Verificar si está instalado (con o sin tag)
                return any(EMBEDDING_MODEL in m for m in models)
    except:
        pass
    return False


def get_embedding_model_name() -> str:
    """Retorna el nombre del modelo de embeddings configurado."""
    return EMBEDDING_MODEL
