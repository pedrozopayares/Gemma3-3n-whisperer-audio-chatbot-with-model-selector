"""
Document Processors - Extrae y procesa texto de diferentes formatos.

Soporta:
- TXT, MD: Texto plano
- DOCX: Microsoft Word
- XLSX, XLS: Microsoft Excel
- PDF: Documentos PDF

Cada procesador retorna chunks de texto con metadata.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Representa un fragmento de documento."""
    content: str
    metadata: Dict
    chunk_id: str


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1000  # caracteres
DEFAULT_OVERLAP = 100


# --------------------------------------------------------------------
# Text Processor (.txt, .md)
# --------------------------------------------------------------------

def process_text_file(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[DocumentChunk]:
    """
    Procesa archivos de texto plano.
    
    Args:
        file_path: Ruta al archivo
        chunk_size: Tamaño máximo de cada chunk
        overlap: Solapamiento entre chunks
    
    Returns:
        Lista de DocumentChunk
    """
    content = file_path.read_text(encoding='utf-8', errors='ignore')
    source = file_path.name
    
    # Dividir por secciones (headers markdown)
    sections = split_by_headers(content)
    
    chunks = []
    chunk_num = 0
    
    for section_title, section_content in sections:
        # Si la sección es pequeña, mantenerla completa
        if len(section_content) <= chunk_size:
            chunks.append(DocumentChunk(
                content=f"## {section_title}\n\n{section_content}" if section_title else section_content,
                metadata={"source": source, "section": section_title, "type": "text"},
                chunk_id=f"{source}_{chunk_num}"
            ))
            chunk_num += 1
        else:
            # Dividir sección grande
            sub_chunks = chunk_text(section_content, chunk_size, overlap)
            for i, sub_chunk in enumerate(sub_chunks):
                header = f"## {section_title} (parte {i+1})\n\n" if section_title else ""
                chunks.append(DocumentChunk(
                    content=header + sub_chunk,
                    metadata={"source": source, "section": section_title, "type": "text", "part": i+1},
                    chunk_id=f"{source}_{chunk_num}"
                ))
                chunk_num += 1
    
    return chunks


def split_by_headers(content: str) -> List[tuple]:
    """Divide contenido por headers markdown."""
    # Pattern para headers: # Header, ## Header, etc.
    pattern = r'^(#{1,6})\s+(.+)$'
    
    sections = []
    current_title = ""
    current_content = []
    
    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            # Guardar sección anterior
            if current_content:
                sections.append((current_title, '\n'.join(current_content).strip()))
            current_title = match.group(2)
            current_content = []
        else:
            current_content.append(line)
    
    # Última sección
    if current_content:
        sections.append((current_title, '\n'.join(current_content).strip()))
    
    # Si no hay headers, retornar todo como una sección
    if not sections:
        sections = [("", content)]
    
    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Divide texto en chunks con solapamiento."""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    # Dividir por párrafos primero
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# --------------------------------------------------------------------
# Word Processor (.docx)
# --------------------------------------------------------------------

def process_word_file(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[DocumentChunk]:
    """
    Procesa archivos Word (.docx).
    
    Args:
        file_path: Ruta al archivo
        chunk_size: Tamaño máximo de cada chunk
    
    Returns:
        Lista de DocumentChunk
    """
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
        from docx.oxml.ns import qn
    except ImportError:
        print("Error: python-docx no instalado. Ejecuta: pip install python-docx")
        return []
    
    doc = Document(str(file_path))
    source = file_path.name
    
    chunks = []
    current_section = "Contenido"
    current_content = []
    chunk_num = 0
    
    def save_current_section():
        nonlocal chunk_num
        if current_content:
            content = '\n'.join(current_content)
            if content.strip():
                # Dividir si es muy grande
                if len(content) > chunk_size:
                    for i, sub in enumerate(chunk_text(content, chunk_size, 100)):
                        chunks.append(DocumentChunk(
                            content=f"## {current_section}\n\n{sub}",
                            metadata={"source": source, "section": current_section, "type": "word"},
                            chunk_id=f"{source}_{chunk_num}"
                        ))
                        chunk_num += 1
                else:
                    chunks.append(DocumentChunk(
                        content=f"## {current_section}\n\n{content}",
                        metadata={"source": source, "section": current_section, "type": "word"},
                        chunk_id=f"{source}_{chunk_num}"
                    ))
                    chunk_num += 1
    
    # Iterar elementos del documento
    for element in doc.element.body.iterchildren():
        if element.tag == qn('w:p'):
            para = Paragraph(element, doc)
            text = para.text.strip()
            
            if not text:
                continue
            
            # Detectar headers por estilo
            if para.style and para.style.name.startswith('Heading'):
                save_current_section()
                current_section = text
                current_content = []
            else:
                current_content.append(text)
        
        elif element.tag == qn('w:tbl'):
            # Convertir tabla a markdown
            table = Table(element, doc)
            table_md = table_to_markdown(table)
            current_content.append(f"\n{table_md}\n")
    
    # Guardar última sección
    save_current_section()
    
    return chunks if chunks else [DocumentChunk(
        content="Documento sin contenido extraíble.",
        metadata={"source": source, "type": "word"},
        chunk_id=f"{source}_0"
    )]


def table_to_markdown(table) -> str:
    """Convierte una tabla de Word a Markdown."""
    rows = []
    for i, row in enumerate(table.rows):
        cells = []
        for cell in row.cells:
            # Limpiar texto de la celda
            cell_text = cell.text.strip().replace('\n', ' ').replace('|', '\\|')
            cells.append(cell_text)
        
        rows.append("| " + " | ".join(cells) + " |")
        
        # Agregar separador después del header
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    
    return "\n".join(rows)


# --------------------------------------------------------------------
# Excel Processor (.xlsx, .xls)
# --------------------------------------------------------------------

def _has_valid_header(df_with_header, df_no_header) -> bool:
    """
    Determina si la primera fila del Excel es un encabezado real de tabla.
    Un encabezado real tiene la mayoría de columnas con nombres descriptivos,
    no valores numéricos ni columnas "Unnamed".
    """
    if df_with_header.empty:
        return False
    cols = [str(c) for c in df_with_header.columns]
    unnamed_count = sum(1 for c in cols if 'unnamed' in c.lower())
    total = len(cols)
    # Si más del 40% de columnas son "Unnamed", no es un header real
    if total > 0 and unnamed_count / total > 0.4:
        return False
    return True


def _excel_to_narrative(df, sheet_name: str) -> str:
    """
    Convierte un DataFrame con estructura no tabular (celdas combinadas,
    formularios, recetas estándar) a texto narrativo legible, eliminando NaN
    y preservando la relación entre celdas adyacentes.
    """
    import pandas as pd
    lines = []
    for _, row in df.iterrows():
        # Recoger solo valores no-nulos de la fila
        values = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
        if values:
            lines.append("  ".join(values))
    return "\n".join(lines)


def process_excel_file(
    file_path: Path,
    rows_per_chunk: int = 30,
    strategy: str = "markdown"  # "markdown" | "rows"
) -> List[DocumentChunk]:
    """
    Procesa archivos Excel.
    
    Detecta automáticamente si la hoja tiene estructura tabular (con headers reales)
    o estructura de formulario/receta (celdas combinadas, sin header claro).
    Para hojas tabulares usa formato markdown; para formularios genera texto narrativo.
    
    Args:
        file_path: Ruta al archivo
        rows_per_chunk: Filas por chunk (para tablas grandes)
        strategy: "markdown" (tabla completa) o "rows" (una entrada por fila)
    
    Returns:
        Lista de DocumentChunk
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas no instalado. Ejecuta: pip install pandas openpyxl")
        return []
    
    source = file_path.name
    chunks = []
    chunk_num = 0
    
    try:
        # Leer todas las hojas
        xls = pd.ExcelFile(file_path)
        
        for sheet_name in xls.sheet_names:
            # Leer con y sin header para detectar estructura
            df_with_header = pd.read_excel(xls, sheet_name=sheet_name)
            df_no_header = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            if df_no_header.empty:
                continue
            
            is_tabular = _has_valid_header(df_with_header, df_no_header)
            
            if not is_tabular:
                # --- Modo narrativo: hojas con celdas combinadas / formularios ---
                narrative = _excel_to_narrative(df_no_header, sheet_name)
                if not narrative.strip():
                    continue
                
                # Dividir en chunks si es muy largo
                if len(narrative) <= DEFAULT_CHUNK_SIZE:
                    chunks.append(DocumentChunk(
                        content=f"## {sheet_name}\n\n{narrative}",
                        metadata={
                            "source": source,
                            "sheet": sheet_name,
                            "type": "excel",
                            "rows": len(df_no_header),
                            "format": "narrative"
                        },
                        chunk_id=f"{source}_{sheet_name}_{chunk_num}"
                    ))
                    chunk_num += 1
                else:
                    text_chunks = chunk_text(narrative, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP)
                    for i, tc in enumerate(text_chunks):
                        chunks.append(DocumentChunk(
                            content=f"## {sheet_name} (parte {i+1})\n\n{tc}",
                            metadata={
                                "source": source,
                                "sheet": sheet_name,
                                "type": "excel",
                                "part": i + 1,
                                "format": "narrative"
                            },
                            chunk_id=f"{source}_{sheet_name}_{chunk_num}"
                        ))
                        chunk_num += 1
                continue
            
            # --- Modo tabular: hojas con headers reales ---
            df = df_with_header
            # Limpiar nombres de columnas
            df.columns = [str(col).strip() for col in df.columns]
            
            if strategy == "markdown":
                # Tabla completa o en chunks
                if len(df) <= rows_per_chunk:
                    md = df.to_markdown(index=False)
                    chunks.append(DocumentChunk(
                        content=f"## {sheet_name}\n\n{md}",
                        metadata={
                            "source": source,
                            "sheet": sheet_name,
                            "type": "excel",
                            "rows": len(df),
                            "columns": list(df.columns)
                        },
                        chunk_id=f"{source}_{sheet_name}_{chunk_num}"
                    ))
                    chunk_num += 1
                else:
                    # Dividir en chunks
                    for i in range(0, len(df), rows_per_chunk):
                        chunk_df = df.iloc[i:i + rows_per_chunk]
                        md = chunk_df.to_markdown(index=False)
                        chunks.append(DocumentChunk(
                            content=f"## {sheet_name} (filas {i+1}-{min(i+rows_per_chunk, len(df))})\n\n{md}",
                            metadata={
                                "source": source,
                                "sheet": sheet_name,
                                "type": "excel",
                                "row_start": i+1,
                                "row_end": min(i+rows_per_chunk, len(df))
                            },
                            chunk_id=f"{source}_{sheet_name}_{chunk_num}"
                        ))
                        chunk_num += 1
            
            elif strategy == "rows":
                # Una entrada por fila
                columns = list(df.columns)
                for idx, row in df.iterrows():
                    parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                    content = f"Registro de {sheet_name}: " + ". ".join(parts) + "."
                    chunks.append(DocumentChunk(
                        content=content,
                        metadata={
                            "source": source,
                            "sheet": sheet_name,
                            "type": "excel_row",
                            "row": int(idx) + 2  # +2 por header y 0-index
                        },
                        chunk_id=f"{source}_{sheet_name}_row{idx}"
                    ))
    
    except Exception as e:
        print(f"Error processing Excel {file_path}: {e}")
        return []
    
    return chunks


# --------------------------------------------------------------------
# PDF Processor (.pdf)
# --------------------------------------------------------------------

def process_pdf_file(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE
) -> List[DocumentChunk]:
    """
    Procesa archivos PDF.
    
    Args:
        file_path: Ruta al archivo
        chunk_size: Tamaño máximo de cada chunk
    
    Returns:
        Lista de DocumentChunk
    """
    try:
        import pypdf
    except ImportError:
        try:
            # Fallback a PyPDF2 para compatibilidad
            import PyPDF2 as pypdf
        except ImportError:
            print("Error: pypdf no instalado. Ejecuta: pip install pypdf")
            return []
    
    source = file_path.name
    chunks = []
    
    try:
        reader = pypdf.PdfReader(str(file_path))
        total_pages = len(reader.pages)
        
        all_text = []
        page_map = []  # Para tracking de páginas
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                cleaned = clean_pdf_text(text)
                all_text.append(cleaned)
                page_map.append((len('\n'.join(all_text)), page_num))
        
        full_text = '\n\n'.join(all_text)
        
        # Dividir en chunks
        text_chunks = chunk_text(full_text, chunk_size, 100)
        
        for i, chunk in enumerate(text_chunks):
            chunks.append(DocumentChunk(
                content=chunk,
                metadata={
                    "source": source,
                    "type": "pdf",
                    "total_pages": total_pages,
                    "chunk": i + 1
                },
                chunk_id=f"{source}_{i}"
            ))
    
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []
    
    return chunks


def clean_pdf_text(text: str) -> str:
    """Limpia texto extraído de PDF."""
    # Remover múltiples espacios
    text = re.sub(r' +', ' ', text)
    # Remover líneas vacías múltiples
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Unir palabras divididas por guión al final de línea
    text = re.sub(r'-\n', '', text)
    return text.strip()


# --------------------------------------------------------------------
# Universal Processor
# --------------------------------------------------------------------

PROCESSORS = {
    '.txt': process_text_file,
    '.md': process_text_file,
    '.docx': process_word_file,
    '.xlsx': process_excel_file,
    '.xls': process_excel_file,
    '.pdf': process_pdf_file,
}

SUPPORTED_EXTENSIONS = set(PROCESSORS.keys())


def process_file(file_path: Path, **kwargs) -> List[DocumentChunk]:
    """
    Procesa cualquier archivo soportado.
    
    Args:
        file_path: Ruta al archivo
        **kwargs: Argumentos adicionales para el procesador
    
    Returns:
        Lista de DocumentChunk
    """
    ext = file_path.suffix.lower()
    
    if ext not in PROCESSORS:
        print(f"Formato no soportado: {ext}")
        return []
    
    processor = PROCESSORS[ext]
    return processor(file_path, **kwargs)


def get_supported_extensions() -> List[str]:
    """Retorna lista de extensiones soportadas."""
    return list(SUPPORTED_EXTENSIONS)
