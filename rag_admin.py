#!/usr/bin/env python3
"""
RAG Admin - CLI para administrar la base de conocimiento.

Uso:
    python rag_admin.py sync              # Sincroniza documents/ con RAG
    python rag_admin.py add archivo.docx  # Agrega un archivo específico
    python rag_admin.py remove fuente     # Elimina documentos de una fuente
    python rag_admin.py list              # Lista documentos indexados
    python rag_admin.py search "consulta" # Prueba una búsqueda
    python rag_admin.py stats             # Muestra estadísticas
    python rag_admin.py rebuild           # Reconstruye toda la base
    python rag_admin.py check             # Verifica dependencias

Ejemplos:
    python rag_admin.py sync
    python rag_admin.py search "¿Cuál es el precio del producto X?"
    python rag_admin.py add documents/manual.docx
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

from rag_module import (
    RAGSystem, 
    load_index, 
    save_index, 
    get_file_hash,
    DOCUMENTS_DIR,
    RAG_DATA_DIR,
    EMBEDDING_MODEL
)
from document_processors import (
    process_file,
    SUPPORTED_EXTENSIONS
)


def print_header(text: str):
    """Imprime un header formateado."""
    print(f"\n{'='*50}")
    print(f"  {text}")
    print(f"{'='*50}")


def print_success(text: str):
    """Imprime mensaje de éxito."""
    print(f"✓ {text}")


def print_error(text: str):
    """Imprime mensaje de error."""
    print(f"✗ {text}")


def print_info(text: str):
    """Imprime mensaje informativo."""
    print(f"ℹ {text}")


# --------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------

def cmd_sync(args):
    """Sincroniza carpeta documents/ con la base de conocimiento."""
    print_header("Sincronizando documentos")
    
    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(parents=True)
        print_info(f"Carpeta {DOCUMENTS_DIR} creada. Agrega documentos y ejecuta sync de nuevo.")
        return
    
    rag = RAGSystem()
    index = load_index()
    
    current_files = {}
    added, updated, removed = 0, 0, 0
    
    # Escanear documents/
    for file_path in DOCUMENTS_DIR.rglob("*"):
        if file_path.is_dir():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name.startswith('.'):
            continue
        # Excluir README.md de la base de conocimiento (es documentación interna)
        if file_path.name.upper() == 'README.MD':
            continue
        
        rel_path = str(file_path.relative_to(DOCUMENTS_DIR))
        file_hash = get_file_hash(file_path)
        current_files[rel_path] = file_hash
        
        # ¿Es nuevo o modificado?
        if rel_path not in index["files"]:
            print(f"  + Agregando: {rel_path}")
            chunks_added = ingest_file(rag, file_path, rel_path)
            if chunks_added > 0:
                added += 1
                index["files"][rel_path] = {
                    "hash": file_hash,
                    "indexed_at": datetime.now().isoformat(),
                    "chunks": chunks_added
                }
        elif index["files"][rel_path]["hash"] != file_hash:
            print(f"  ~ Actualizando: {rel_path}")
            # Eliminar chunks anteriores
            rag.delete_by_source(rel_path)
            # Reingestar
            chunks_added = ingest_file(rag, file_path, rel_path)
            if chunks_added > 0:
                updated += 1
                index["files"][rel_path] = {
                    "hash": file_hash,
                    "indexed_at": datetime.now().isoformat(),
                    "chunks": chunks_added
                }
    
    # Detectar archivos eliminados
    for rel_path in list(index["files"].keys()):
        if rel_path not in current_files:
            print(f"  - Eliminando: {rel_path}")
            rag.delete_by_source(rel_path)
            del index["files"][rel_path]
            removed += 1
    
    # Guardar índice
    index["last_sync"] = datetime.now().isoformat()
    save_index(index)
    
    print(f"\n{'─'*40}")
    print_success(f"Sincronización completa:")
    print(f"  Agregados:    {added}")
    print(f"  Actualizados: {updated}")
    print(f"  Eliminados:   {removed}")
    print(f"  Total docs:   {len(index['files'])}")
    print(f"  Total chunks: {rag.get_stats()['total_documents']}")


def cmd_add(args):
    """Agrega un archivo específico."""
    file_path = Path(args.file)
    
    if not file_path.exists():
        print_error(f"Archivo no encontrado: {file_path}")
        return
    
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print_error(f"Formato no soportado: {file_path.suffix}")
        print_info(f"Formatos soportados: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    print_header(f"Agregando: {file_path.name}")
    
    rag = RAGSystem()
    index = load_index()
    
    rel_path = file_path.name
    chunks_added = ingest_file(rag, file_path, rel_path)
    
    if chunks_added > 0:
        index["files"][rel_path] = {
            "hash": get_file_hash(file_path),
            "indexed_at": datetime.now().isoformat(),
            "chunks": chunks_added
        }
        save_index(index)
        print_success(f"Agregado: {chunks_added} chunks indexados")
    else:
        print_error("No se pudieron extraer chunks del archivo")


def cmd_remove(args):
    """Elimina documentos de una fuente."""
    source = args.source
    
    print_header(f"Eliminando: {source}")
    
    rag = RAGSystem()
    index = load_index()
    
    # Buscar coincidencias
    matches = [k for k in index["files"].keys() if source in k]
    
    if not matches:
        print_error(f"No se encontró: {source}")
        return
    
    for match in matches:
        deleted = rag.delete_by_source(match)
        del index["files"][match]
        print_success(f"Eliminado: {match} ({deleted} chunks)")
    
    save_index(index)


def cmd_list(args):
    """Lista documentos indexados."""
    index = load_index()
    
    if not index["files"]:
        print_info("No hay documentos indexados.")
        print_info(f"Coloca archivos en {DOCUMENTS_DIR}/ y ejecuta: python rag_admin.py sync")
        return
    
    print_header(f"Documentos indexados ({len(index['files'])})")
    
    for rel_path, info in sorted(index["files"].items()):
        chunks = info.get('chunks', '?')
        indexed_at = info.get('indexed_at', '')[:10]
        print(f"  📄 {rel_path}")
        print(f"     Chunks: {chunks} | Indexado: {indexed_at}")


def cmd_search(args):
    """Prueba una búsqueda."""
    query = " ".join(args.query)
    
    if not query:
        print_error("Especifica una consulta")
        return
    
    print_header(f"Buscando: {query}")
    
    rag = RAGSystem()
    results = rag.search(query, n_results=args.n or 3)
    
    if not results:
        print_info("Sin resultados")
        return
    
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "Desconocido")
        relevance = result.get("relevance", 0) * 100
        content = result["content"][:300] + "..." if len(result["content"]) > 300 else result["content"]
        
        print(f"\n{i}. [{source}] (relevancia: {relevance:.1f}%)")
        print(f"   {content}")


def cmd_stats(args):
    """Muestra estadísticas."""
    rag = RAGSystem()
    index = load_index()
    stats = rag.get_stats()
    
    print_header("Estadísticas RAG")
    
    print(f"  Documentos fuente: {len(index['files'])}")
    print(f"  Chunks indexados:  {stats['total_documents']}")
    print(f"  Modelo embeddings: {stats['embedding_model']}")
    print(f"  Última sync:       {index.get('last_sync', 'Nunca')}")
    
    # Tamaño en disco
    if RAG_DATA_DIR.exists():
        size = sum(f.stat().st_size for f in RAG_DATA_DIR.rglob("*") if f.is_file())
        print(f"  Tamaño en disco:   {size / 1024 / 1024:.2f} MB")
    
    # Por tipo de archivo
    print(f"\n  Por tipo:")
    by_type = {}
    for rel_path in index["files"]:
        ext = Path(rel_path).suffix.lower()
        by_type[ext] = by_type.get(ext, 0) + 1
    
    for ext, count in sorted(by_type.items()):
        print(f"    {ext}: {count}")


def cmd_rebuild(args):
    """Reconstruye toda la base de conocimiento."""
    print_header("Reconstruir base de conocimiento")
    
    print("⚠️  Esto eliminará toda la base y la reconstruirá desde cero.")
    
    if not args.force:
        confirm = input("¿Continuar? [s/N]: ")
        if confirm.lower() != 's':
            print("Cancelado.")
            return
    
    # Eliminar base existente
    if RAG_DATA_DIR.exists():
        shutil.rmtree(RAG_DATA_DIR)
        print_info("Base eliminada")
    
    print_info("Reconstruyendo...")
    
    # Crear argumentos fake para sync
    class SyncArgs:
        pass
    
    cmd_sync(SyncArgs())


def cmd_check(args):
    """Verifica dependencias y configuración."""
    print_header("Verificando dependencias")
    
    all_ok = True
    
    # ChromaDB
    try:
        import chromadb
        print_success(f"chromadb: {chromadb.__version__}")
    except ImportError:
        print_error("chromadb: NO INSTALADO - pip install chromadb")
        all_ok = False
    
    # python-docx
    try:
        import docx
        print_success("python-docx: OK")
    except ImportError:
        print_error("python-docx: NO INSTALADO - pip install python-docx")
        all_ok = False
    
    # pandas
    try:
        import pandas
        print_success(f"pandas: {pandas.__version__}")
    except ImportError:
        print_error("pandas: NO INSTALADO - pip install pandas openpyxl")
        all_ok = False
    
    # openpyxl
    try:
        import openpyxl
        print_success(f"openpyxl: {openpyxl.__version__}")
    except ImportError:
        print_error("openpyxl: NO INSTALADO - pip install openpyxl")
        all_ok = False
    
    # pypdf
    try:
        import pypdf
        print_success(f"pypdf: {pypdf.__version__}")
    except ImportError:
        print_error("pypdf: NO INSTALADO - pip install pypdf")
        all_ok = False
    
    # tabulate (para markdown tables)
    try:
        import tabulate
        print_success("tabulate: OK")
    except ImportError:
        print_error("tabulate: NO INSTALADO - pip install tabulate")
        all_ok = False
    
    # httpx
    try:
        import httpx
        print_success(f"httpx: OK")
    except ImportError:
        print_error("httpx: NO INSTALADO - pip install httpx")
        all_ok = False
    
    # Verificar Ollama
    print(f"\n{'─'*40}")
    print("Ollama:")
    
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            print_success("Ollama está corriendo")
            
            # Verificar modelo de embeddings
            if any(EMBEDDING_MODEL in m for m in models):
                print_success(f"Modelo {EMBEDDING_MODEL} disponible")
            else:
                print_error(f"Modelo {EMBEDDING_MODEL} NO INSTALADO")
                print_info(f"  Ejecuta: ollama pull {EMBEDDING_MODEL}")
                all_ok = False
        else:
            print_error("Ollama no responde")
            all_ok = False
    except Exception as e:
        print_error(f"No se puede conectar a Ollama: {e}")
        print_info("  Ejecuta: ollama serve")
        all_ok = False
    
    # Verificar carpeta documents
    print(f"\n{'─'*40}")
    print("Estructura:")
    
    if DOCUMENTS_DIR.exists():
        print_success(f"Carpeta {DOCUMENTS_DIR} existe")
    else:
        print_info(f"Carpeta {DOCUMENTS_DIR} no existe (se creará en sync)")
    
    print(f"\n{'─'*40}")
    if all_ok:
        print_success("Todo listo para usar RAG")
    else:
        print_error("Hay dependencias faltantes")
        print_info("Instala todo con: pip install chromadb python-docx pandas openpyxl pypdf tabulate")


# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def ingest_file(rag: RAGSystem, file_path: Path, rel_path: str) -> int:
    """
    Ingesta un archivo en la base de conocimiento.
    
    Returns:
        Número de chunks agregados
    """
    try:
        chunks = process_file(file_path)
        
        if not chunks:
            return 0
        
        documents = [c.content for c in chunks]
        ids = [f"{rel_path}_{c.chunk_id}" for c in chunks]
        metadatas = [{**c.metadata, "source": rel_path} for c in chunks]
        
        return rag.add_documents(documents, ids, metadatas)
    
    except Exception as e:
        print_error(f"Error procesando {file_path}: {e}")
        return 0


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Admin - Administrador de base de conocimiento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python rag_admin.py check              # Verificar dependencias
  python rag_admin.py sync               # Sincronizar documents/
  python rag_admin.py search "precio"    # Buscar información
  python rag_admin.py stats              # Ver estadísticas
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # sync
    p_sync = subparsers.add_parser('sync', help='Sincroniza documents/ con RAG')
    
    # add
    p_add = subparsers.add_parser('add', help='Agrega un archivo')
    p_add.add_argument('file', help='Ruta al archivo')
    
    # remove
    p_remove = subparsers.add_parser('remove', help='Elimina documentos')
    p_remove.add_argument('source', help='Nombre del archivo fuente')
    
    # list
    p_list = subparsers.add_parser('list', help='Lista documentos indexados')
    
    # search
    p_search = subparsers.add_parser('search', help='Prueba una búsqueda')
    p_search.add_argument('query', nargs='+', help='Consulta de búsqueda')
    p_search.add_argument('-n', type=int, default=3, help='Número de resultados')
    
    # stats
    p_stats = subparsers.add_parser('stats', help='Muestra estadísticas')
    
    # rebuild
    p_rebuild = subparsers.add_parser('rebuild', help='Reconstruye la base')
    p_rebuild.add_argument('--force', '-f', action='store_true', help='No pedir confirmación')
    
    # check
    p_check = subparsers.add_parser('check', help='Verifica dependencias')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        'sync': cmd_sync,
        'add': cmd_add,
        'remove': cmd_remove,
        'list': cmd_list,
        'search': cmd_search,
        'stats': cmd_stats,
        'rebuild': cmd_rebuild,
        'check': cmd_check,
    }
    
    commands[args.command](args)


if __name__ == "__main__":
    main()
