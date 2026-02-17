# 📚 Carpeta de Documentos para RAG

Coloca aquí los archivos que quieres incluir en la base de conocimiento del asistente.

## Formatos Soportados

| Formato | Extensiones | Notas |
|---------|-------------|-------|
| **Texto** | `.txt`, `.md` | Procesado por secciones (headers) |
| **Word** | `.docx` | Extrae texto y tablas |
| **Excel** | `.xlsx`, `.xls` | Convierte tablas a formato legible |
| **PDF** | `.pdf` | Extrae texto de todas las páginas |

## Cómo Usar

### 1. Agregar documentos
Simplemente copia tus archivos a esta carpeta o subcarpetas:

```
documents/
├── manuales/
│   └── manual_producto.docx
├── productos/
│   └── catalogo.xlsx
└── faqs/
    └── preguntas_frecuentes.md
```

### 2. Sincronizar con RAG
Ejecuta en la terminal:

```bash
python rag_admin.py sync
```

### 3. Verificar
```bash
python rag_admin.py list    # Ver documentos indexados
python rag_admin.py stats   # Ver estadísticas
```

### 4. Probar búsqueda
```bash
python rag_admin.py search "mi pregunta"
```

## Comandos Útiles

| Comando | Descripción |
|---------|-------------|
| `python rag_admin.py sync` | Sincroniza esta carpeta con RAG |
| `python rag_admin.py list` | Lista documentos indexados |
| `python rag_admin.py stats` | Muestra estadísticas |
| `python rag_admin.py search "texto"` | Prueba una búsqueda |
| `python rag_admin.py rebuild` | Reconstruye la base completa |
| `python rag_admin.py check` | Verifica dependencias |

## Actualizaciones

- **Modificar documento**: Reemplaza el archivo y ejecuta `sync`
- **Eliminar documento**: Borra el archivo y ejecuta `sync`
- **Agregar documento**: Copia el archivo y ejecuta `sync`

## Tips

1. **Organiza por carpetas** - Ayuda a mantener el orden
2. **Nombres descriptivos** - Aparecen como fuente en las respuestas
3. **Documentos pequeños** - Se procesan más rápido
4. **Excel con headers claros** - Mejora la búsqueda semántica
