# Informe Técnico: Arquitectura RAG del Chatbot Conversacional con IA Local

**Proyecto:** Gemma3-3n Whisperer — Audio Chatbot with Model Selector  
**Versión:** MVP 1.0  
**Fecha:** Febrero 2026  
**Alcance:** Documentación técnica completa del marco RAG (Retrieval-Augmented Generation) integrado al chatbot conversacional multimodal

---

## 1. Resumen Ejecutivo

Este documento describe la arquitectura, el diseño, la implementación y el funcionamiento del sistema de Generación Aumentada por Recuperación (RAG) integrado en un chatbot conversacional que opera completamente en infraestructura local. El sistema permite que los modelos de lenguaje (LLM) respondan preguntas fundamentadas en documentos corporativos reales, eliminando alucinaciones sobre temas específicos del dominio y priorizando el conocimiento verificado sobre la generación especulativa.

El MVP implementa un pipeline completo que abarca desde la ingesta de documentos en múltiples formatos (.docx, .xlsx, .pdf, .txt, .md) hasta la presentación de resultados enriquecidos con fuentes citadas en la interfaz de usuario. Todo el procesamiento ocurre localmente sin dependencias de servicios cloud, lo cual garantiza la privacidad de los datos corporativos y elimina costos recurrentes de APIs externas.

La arquitectura resultante combina cinco subsistemas interconectados: procesamiento de documentos con chunking inteligente, indexación vectorial persistente con ChromaDB, generación de embeddings semánticos con Ollama, enrutamiento inteligente de consultas entre modelos especializados, y una interfaz React que expone las fuentes de conocimiento utilizadas para cada respuesta.

---

## 2. Fundamentos y Justificación del Enfoque RAG

### 2.1 Problema que Resuelve

Los modelos de lenguaje grandes, aun cuando son capaces de generar texto coherente y contextualmente plausible, presentan una limitación fundamental: su conocimiento está congelado en el momento del entrenamiento. Cuando un usuario pregunta por procedimientos internos, recetas específicas o protocolos operativos que nunca formaron parte de los datos de entrenamiento, el modelo tiene dos opciones: admitir desconocimiento o fabricar una respuesta verosímil pero incorrecta. En entornos productivos donde la precisión es crítica — como la estandarización de recetas en un restaurante — esta segunda opción resulta inaceptable.

RAG resuelve este problema inyectando conocimiento externo verificado directamente en el contexto del modelo antes de que genere su respuesta. En lugar de depender exclusivamente de los parámetros aprendidos durante el entrenamiento, el modelo recibe fragmentos relevantes de documentos corporativos como parte de su prompt, lo que le permite fundamentar sus respuestas en información real y citar las fuentes utilizadas.

### 2.2 Por Qué RAG y No Fine-Tuning

La alternativa clásica a RAG es el fine-tuning (ajuste fino) del modelo con datos propios. Se descartó esta opción por tres razones concretas. Primero, el fine-tuning requiere conjuntos de datos de entrenamiento curados en formato pregunta-respuesta, lo cual demanda un esfuerzo significativo de preparación que no escala cuando los documentos cambian frecuentemente. Segundo, cada actualización de contenido requeriría reentrenar el modelo, un proceso que consume horas de cómputo GPU y genera una nueva versión del modelo que debe desplegarse. Tercero, RAG preserva la trazabilidad: cada respuesta puede acompañarse de la fuente documental exacta que la sustenta, algo que el fine-tuning no ofrece porque el conocimiento se diluye en los pesos del modelo.

### 2.3 Decisión de Infraestructura Local

Toda la infraestructura opera localmente mediante Ollama para inferencia LLM y embeddings, ChromaDB como base de datos vectorial embebida, y Whisper para transcripción de audio. Esta decisión responde a tres factores: la naturaleza confidencial de los documentos corporativos que no deben transitar por APIs externas, la eliminación de costos recurrentes por token que generan servicios como OpenAI o Anthropic, y la capacidad de operar sin conexión a internet una vez que los modelos están descargados.

---

## 3. Arquitectura del Sistema

### 3.1 Vista General

El sistema sigue una arquitectura de tres capas con un flujo de datos unidireccional desde la interfaz de usuario hasta los modelos de IA, enriquecido en cada paso con contexto recuperado de la base de conocimiento.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                         │
│  React 19 + Vite 7 + TailwindCSS 4                             │
│  Audio Recording ─ Text Input ─ Image Upload ─ TTS Playback    │
│  Renderizado Markdown ─ Panel de Fuentes RAG colapsable         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/JSON (localhost:5173 → :8000)
┌──────────────────────────▼──────────────────────────────────────┐
│                    CAPA DE APLICACIÓN                           │
│  FastAPI (gemma_server.py)                                      │
│  ┌──────────┐  ┌──────────────┐  ┌─────────────┐               │
│  │ Whisper  │  │ Smart Router │  │ RAG Search  │               │
│  │ STT      │  │ qwen2.5:0.5b │  │ ChromaDB   │               │
│  └────┬─────┘  └──────┬───────┘  └──────┬──────┘               │
│       │               │                 │                       │
│       └───────────────▼─────────────────┘                       │
│           ┌─────────────────────────┐                           │
│           │   Ollama LLM Backend    │                           │
│           │ gemma3:4b ── phi4:latest│                           │
│           └─────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    CAPA DE DATOS                                │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────────┐        │
│  │ ChromaDB    │  │ .index.json   │  │ documents/     │        │
│  │ (rag_data/) │  │ File Tracking │  │ Source Files   │        │
│  │ SQLite+HNSW │  │ Hash Registry │  │ .docx .xlsx .md│        │
│  └─────────────┘  └───────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Componentes y Responsabilidades

El sistema se compone de seis módulos principales, cada uno con un ámbito de responsabilidad claramente delimitado.

**`document_processors.py`** se encarga exclusivamente de la extracción y fragmentación de texto desde archivos fuente. Implementa procesadores especializados por formato (Word, Excel, PDF, texto plano y Markdown) y aplica estrategias de chunking adaptadas a la estructura de cada tipo de documento. Su salida son objetos `DocumentChunk` que encapsulan el contenido textual junto con metadatos de origen.

**`rag_module.py`** constituye el núcleo del sistema RAG. Define la clase `OllamaEmbeddingFunction` que adapta la API de embeddings de Ollama al protocolo que ChromaDB requiere, la clase `RAGSystem` que abstrae todas las operaciones de la base de datos vectorial, y funciones auxiliares para gestión de índices y construcción de prompts enriquecidos con contexto.

**`rag_admin.py`** proporciona una interfaz de línea de comandos (CLI) para administrar la base de conocimiento. Permite sincronizar documentos, agregar o eliminar fuentes, buscar contenido, visualizar estadísticas y reconstruir el índice completo. Opera como herramienta de mantenimiento independiente del servidor web.

**`gemma_server.py`** es el servidor FastAPI que orquesta todo el flujo: recibe peticiones HTTP del frontend, transcribe audio con Whisper cuando corresponde, consulta la base de conocimiento RAG, construye prompts enriquecidos con contexto documental, enruta consultas al modelo apropiado mediante clasificación automática, y retorna respuestas junto con los fragmentos de fuentes utilizados.

**`gemma-chatbot-ui/src/App.jsx`** implementa la interfaz de usuario completa en React. Gestiona grabación de audio, entrada de texto, carga de imágenes, selección de modelos, visualización de conversaciones con renderizado Markdown, reproducción TTS, y un panel colapsable que muestra las fuentes RAG asociadas a cada respuesta del modelo.

**`rag_data/`** almacena de forma persistente la base de datos vectorial ChromaDB (SQLite + índices HNSW) y el archivo de tracking `.index.json` que registra los hashes MD5 de cada archivo indexado para detección incremental de cambios.

### 3.3 Flujo de Datos Completo

Cuando un usuario envía una consulta — ya sea por voz o texto — el sistema ejecuta la siguiente secuencia:

1. Si la entrada es audio, Whisper la transcribe a texto en español con parámetros optimizados (temperatura 0, sin condicionamiento sobre texto previo, deshabilitando FP16 para mayor precisión).

2. El texto de la consulta se envía al módulo RAG, que genera un embedding vectorial de 768 dimensiones utilizando el modelo `nomic-embed-text` a través de la API `/api/embed` de Ollama.

3. ChromaDB ejecuta una búsqueda por similitud coseno (HNSW) contra los embeddings almacenados de los chunks documentales, retornando los `RAG_TOP_K` resultados más cercanos (por defecto 3).

4. Los resultados se filtran por relevancia mínima (`RAG_MIN_RELEVANCE = 0.3`). Solo los fragmentos que superan el umbral se incorporan al contexto.

5. Si el modo de enrutamiento es "auto", el modelo ligero `qwen2.5:0.5b` clasifica la consulta como "math" o "chat" para dirigirla al modelo especializado correspondiente (`phi4:latest` para matemáticas, `gemma3:4b` para conversación general).

6. Se construye el system prompt combinando el contexto interno del asistente, los fragmentos RAG relevantes (marcados como "PRIORIDAD ALTA"), el resumen de conversación anterior si existe, y el contexto personalizado del usuario desde el frontend.

7. El modelo LLM genera la respuesta fundamentada en el contexto documental inyectado.

8. La respuesta se retorna al frontend junto con los `rag_chunks` (fuente, sección, relevancia y contenido truncado a 500 caracteres), que se renderizan en un panel colapsable bajo cada mensaje del asistente.

---

## 4. Pipeline de Procesamiento de Documentos

### 4.1 Formatos Soportados y Estrategias de Extracción

El módulo `document_processors.py` implementa procesadores especializados para cinco familias de formatos de archivo, registrados en el diccionario `PROCESSORS` que mapea extensiones a funciones de procesamiento.

**Archivos de texto plano y Markdown (.txt, .md):** La función `process_text_file` detecta la estructura del documento analizando encabezados Markdown con la expresión regular `^(#{1,6})\s+(.+)$`. Cada sección definida por un encabezado se convierte en un chunk independiente si su longitud no excede `DEFAULT_CHUNK_SIZE` (1000 caracteres). Las secciones más extensas se subdividen mediante `chunk_text`, que prioriza la división por párrafos dobles (`\n\n`) para preservar la coherencia semántica de cada fragmento. Los chunks resultantes se prefijan con el encabezado de la sección original para mantener el contexto estructural.

**Documentos Word (.docx):** La función `process_word_file` utiliza la biblioteca `python-docx` para iterar sobre los elementos XML del cuerpo del documento. Detecta encabezados mediante el estilo del párrafo (`para.style.name.startswith('Heading')`) y agrupa el contenido subsecuente bajo cada sección. Las tablas encontradas en el documento se convierten a formato Markdown mediante `table_to_markdown`, que genera una representación textual con separadores de columnas (`|`) y línea de encabezado (`|---|`), facilitando la comprensión por parte del modelo de lenguaje.

**Hojas de cálculo Excel (.xlsx, .xls):** La función `process_excel_file` emplea `pandas` con el motor `openpyxl` para leer cada hoja del documento. Soporta dos estrategias de chunking configurables: el modo "markdown" convierte la tabla completa (o fragmentos de `rows_per_chunk` filas) a formato Markdown tabulado mediante `DataFrame.to_markdown()`, mientras que el modo "rows" genera un chunk narrativo por cada fila con el patrón "Columna: Valor. Columna: Valor.", lo cual resulta más adecuado para tablas de registros independientes como inventarios o listas de ingredientes.

**Documentos PDF (.pdf):** La función `process_pdf_file` utiliza `pypdf` (con fallback a `PyPDF2` para compatibilidad) para extraer texto página por página. El texto extraído se limpia mediante `clean_pdf_text`, que normaliza espacios múltiples, consolida saltos de línea consecutivos y reconecta palabras divididas por guiones al final de línea (`-\n` → concatenación directa). Los metadatos incluyen el número total de páginas y la posición del chunk dentro del documento.

### 4.2 Estrategia de Chunking

El chunking es una de las decisiones de diseño más críticas en un sistema RAG porque determina la granularidad de la información recuperable. Chunks demasiado grandes diluyen la relevancia semántica y consumen tokens de contexto innecesariamente; chunks demasiado pequeños pierden coherencia y dificultan que el modelo comprenda el contexto completo de una instrucción o procedimiento.

La implementación actual utiliza un enfoque híbrido con dos niveles. En el primer nivel, el documento se divide por su estructura natural: encabezados en archivos de texto y Word, hojas en archivos Excel, y páginas en archivos PDF. En el segundo nivel, cada sección resultante se evalúa contra el tamaño máximo configurado (`DEFAULT_CHUNK_SIZE = 1000` caracteres, aproximadamente 250-300 tokens). Las secciones que exceden este límite se subdividen por párrafos dobles, con un solapamiento configurable de `DEFAULT_OVERLAP = 100` caracteres entre chunks consecutivos para evitar la pérdida de contexto en las fronteras.

Cada chunk generado se encapsula en un objeto `DocumentChunk` con tres atributos: `content` (el texto fragmentado, prefijado con el encabezado de sección), `metadata` (un diccionario con el archivo fuente, la sección, el tipo de procesador y atributos específicos del formato), y `chunk_id` (un identificador único construido como `{nombre_archivo}_{número_secuencial}`).

### 4.3 Modelo de Datos del Chunk

La estructura de metadatos varía según el procesador pero mantiene un esquema base común:

| Campo        | Tipo   | Descripción                                  | Presente en     |
|-------------|--------|----------------------------------------------|-----------------|
| `source`     | str    | Ruta relativa del archivo fuente             | Todos           |
| `section`    | str    | Encabezado o nombre de sección               | Todos           |
| `type`       | str    | Tipo de procesador (text, word, excel, pdf)  | Todos           |
| `indexed_at` | str    | Timestamp ISO 8601 de indexación             | Post-ingesta    |
| `sheet`      | str    | Nombre de la hoja Excel                      | Excel           |
| `rows`       | int    | Cantidad de filas en el chunk                | Excel           |
| `columns`    | list   | Nombres de columnas                          | Excel           |
| `total_pages`| int    | Total de páginas del PDF                     | PDF             |
| `chunk`      | int    | Número de chunk dentro del PDF               | PDF             |
| `part`       | int    | Número de parte cuando se subdivide sección  | Texto/Word      |

---

## 5. Motor de Embeddings y Base Vectorial

### 5.1 Modelo de Embeddings: nomic-embed-text

El sistema utiliza `nomic-embed-text` como modelo de embeddings, ejecutado localmente a través de Ollama. Este modelo genera vectores de 768 dimensiones que capturan la semántica del texto en un espacio donde la proximidad entre vectores refleja la similitud de significado entre los textos que representan.

La elección de `nomic-embed-text` se fundamenta en su equilibrio entre calidad y eficiencia: ocupa aproximadamente 275 MB de almacenamiento, genera embeddings en milisegundos para textos cortos, y produce representaciones semánticas competitivas con modelos significativamente más grandes. Al ejecutarse localmente mediante Ollama, elimina la latencia de red y los costos por token que implicaría utilizar servicios como la API de embeddings de OpenAI.

### 5.2 Adaptador OllamaEmbeddingFunction

La clase `OllamaEmbeddingFunction` actúa como adaptador entre la API REST de Ollama y el protocolo de funciones de embedding que ChromaDB requiere. Este adaptador fue uno de los componentes que requirió mayor iteración durante el desarrollo debido a incompatibilidades entre las interfaces.

ChromaDB 1.5 define el protocolo `EmbeddingFunction[D]` que requiere que el método `__call__` acepte un parámetro `input` de tipo genérico `D` y retorne `List[np.ndarray]` (alias `Embeddings`). Adicionalmente, expone los métodos `embed_query` y `embed_documents` para diferenciar el procesamiento de consultas y documentos.

El desafío técnico principal surgió de que ChromaDB 1.5 pasa el input a `embed_query` como una **lista** (e.g., `input=['texto de búsqueda']`) en lugar de como una cadena simple, y que `__call__` recibe los textos como listas anidadas (e.g., `[['texto1'], ['texto2']]`). Simultáneamente, la API de Ollama cambió su endpoint de `/api/embeddings` (con payload `{"prompt": text}`) a `/api/embed` (con payload `{"input": text_or_list}`), y el formato de respuesta de `data["embedding"]` a `data["embeddings"]`. El resultado retornado debe ser `List[np.ndarray]`, no `List[List[float]]`, ya que ChromaDB 1.5 valida los tipos estrictamente.

La implementación final resuelve estas incompatibilidades de la siguiente manera:

```python
def __call__(self, input) -> List[np.ndarray]:
    # Flatten listas anidadas de ChromaDB 1.5: [['text1']] → ['text1']
    flat_texts = []
    if input and isinstance(input[0], list):
        for item in input:
            flat_texts.append(item[0] if item else "")
    else:
        flat_texts = list(input)
    raw = self._get_embeddings(flat_texts)
    return [np.array(e, dtype=np.float32) for e in raw]

def embed_query(self, input="", query="", **kwargs) -> List[np.ndarray]:
    # ChromaDB 1.5 pasa input como lista: embed_query(input=['text'])
    if isinstance(input, list):
        texts = [t if isinstance(t, str) else str(t) for t in input]
    else:
        texts = [input or query]
    raw = self._get_embeddings(texts)
    return [np.array(e, dtype=np.float32) for e in raw]
```

El método interno `_get_embeddings` centraliza la comunicación HTTP con Ollama, enviando las listas de texto ya normalizadas al endpoint `/api/embed` y retornando los vectores crudos. Implementa un mecanismo de fallback que produce vectores nulos de 768 dimensiones (`[0.0] * 768`) cuando el servicio no está disponible, evitando que errores de conectividad colapsen el pipeline completo.

### 5.3 ChromaDB: Base de Datos Vectorial

ChromaDB opera como la base de datos vectorial del sistema, configurada en modo persistente con almacenamiento en el directorio `rag_data/`. La persistencia se logra mediante un backend SQLite que almacena los documentos, metadatos e identificadores, complementado con índices HNSW (Hierarchical Navigable Small World) que permiten búsquedas aproximadas de vecinos más cercanos en tiempo sublineal.

La colección se configura con espacio métrico de similitud coseno (`"hnsw:space": "cosine"`), que mide la similitud entre vectores por el ángulo que forman en lugar de la distancia euclidiana. Esta métrica es estándar para embeddings de texto porque normaliza la magnitud de los vectores, concentrándose exclusivamente en la dirección — es decir, en la semántica — de las representaciones.

La clase `RAGSystem` encapsula todas las operaciones sobre ChromaDB:

La **inicialización** crea el directorio de datos si no existe, instancia un `PersistentClient` con telemetría deshabilitada, configura la función de embeddings, y obtiene o crea la colección con nombre `knowledge_base`.

La **adición de documentos** acepta listas paralelas de textos, identificadores únicos y metadatos opcionales. Si no se proporcionan metadatos, genera automáticamente un diccionario con la marca temporal de indexación.

La **búsqueda** invoca `collection.query` con `query_texts` (delegando la generación de embeddings a ChromaDB internamente), retorna los `n_results` documentos más cercanos, y formatea los resultados añadiendo un campo `relevance` calculado como `1 - distance` (dado que la distancia coseno oscila entre 0 —idénticos— y 2 —opuestos—, la relevancia resultante varía entre -1 y 1, aunque en la práctica los valores para documentos relacionados caen entre 0.3 y 0.8).

La **eliminación** soporta borrado por fuente (`delete_by_source`, que busca por metadata y elimina todos los chunks asociados a un archivo) y por identificadores específicos (`delete_by_ids`).

El **vaciado completo** (`clear`) elimina la colección y la recrea, lo cual es necesario durante reconstrucciones del índice.

### 5.4 Gestión de Índice Incremental

El archivo `.index.json` mantiene un registro de todos los archivos procesados con sus hashes MD5 y la cantidad de chunks generados. Este mecanismo permite que la sincronización sea incremental: al ejecutar `rag_admin.py sync`, el sistema compara el hash actual de cada archivo contra el hash registrado. Solo los archivos nuevos (no presentes en el índice) o modificados (hash diferente) se reprocesan. Los archivos que ya no existen en el directorio `documents/` se eliminan automáticamente de la base de conocimiento. Esta estrategia evita reprocesar colecciones documentales completas cuando solo cambia un archivo, reduciendo significativamente el tiempo de actualización.

---

## 6. Integración RAG en el Servidor Backend

### 6.1 Inicialización Lazy del Sistema RAG

El servidor inicializa el sistema RAG de forma diferida (lazy loading) mediante la función `get_rag_system()`. La primera invocación instancia `RAGSystem` y almacena la referencia en una variable global `_rag_system`. Invocaciones subsecuentes retornan la instancia existente sin incurrir en el costo de reconexión a ChromaDB. Si la inicialización falla (por ejemplo, si ChromaDB no está instalado), el sistema continúa operando sin RAG, degradando la funcionalidad de forma elegante en lugar de fallar catastróficamente.

La importación del módulo RAG también está protegida por un bloque try/except que establece el flag `RAG_AVAILABLE` a `False` si las dependencias no están instaladas, permitiendo que el servidor funcione como chatbot puro sin base de conocimiento.

### 6.2 Búsqueda y Filtrado de Contexto

La función asíncrona `search_rag_context` ejecuta la búsqueda RAG y filtra los resultados por relevancia mínima:

```python
async def search_rag_context(query: str) -> tuple[str, List[Dict]]:
    results = rag.search(query, n_results=RAG_TOP_K)  # Top 3
    relevant = [r for r in results if r.get("relevance", 0) >= RAG_MIN_RELEVANCE]  # >= 0.3
```

Los documentos que superan el umbral se formatean como texto estructurado con encabezados que incluyen la fuente, sección y porcentaje de relevancia. Este texto formateado se inyecta en el system prompt, mientras que los resultados crudos se retornan al frontend como `rag_chunks` para visualización.

### 6.3 Construcción del System Prompt Enriquecido

La función `build_system_prompt` compone el prompt del sistema mediante capas jerárquicas de contexto. La capa base es `INTERNAL_CONTEXT`, un prompt estático que define la personalidad y las directrices generales del asistente. Sobre esta base se apilan, en orden de prioridad: el contexto RAG (documentos relevantes), el resumen de conversación anterior (si la historia excede 20 mensajes), y el contexto personalizado desde el frontend.

El contexto RAG se marca explícitamente con la instrucción "PRIORIDAD ALTA" y directivas claras para que el modelo base sus respuestas en los documentos cuando sean relevantes, complemente con conocimiento general solo cuando los documentos no cubran la pregunta, y cite la fuente cuando utilice información de la base de conocimiento. Esta jerarquización explícita ha demostrado ser necesaria porque los modelos tienden a favorecer su conocimiento paramétrico sobre el contexto inyectado cuando las instrucciones no son suficientemente directivas.

### 6.4 Enrutamiento Inteligente de Consultas

El sistema implementa un mecanismo de enrutamiento que clasifica automáticamente cada consulta para dirigirla al modelo más adecuado. El modelo ultraligero `qwen2.5:0.5b` (~270 MB) opera como clasificador con temperatura 0 (determinístico) y predicción limitada a 10 tokens. El prompt de clasificación le pide que responda con una sola palabra: "math" para consultas que requieren cálculos, conversiones o proporciones, y "chat" para consultas conversacionales o informativas.

Las consultas matemáticas se dirigen a `phi4:latest` (14B parámetros, razonamiento avanzado), mientras que las conversacionales van a `gemma3:4b` (balance calidad/tamaño). El usuario también puede seleccionar manualmente un modelo específico o deshabilitar el enrutamiento automático.

Este mecanismo es particularmente relevante para el caso de uso RAG en gastronomía: una pregunta como "¿cuántos gramos de sal necesito para 5 porciones de pollo juanillo?" involucra tanto la recuperación de la receta (RAG + chat) como un cálculo proporcional (math), y el sistema puede enrutar la respuesta al modelo que mejor maneje la aritmética implicada.

### 6.5 Endpoints RAG Expuestos

El servidor expone cinco endpoints dedicados a la administración RAG:

`GET /rag/status` retorna el estado del sistema incluyendo si está habilitado, inicializado, el conteo de documentos y chunks, el modelo de embeddings, la fecha de última sincronización y los parámetros de configuración actuales (top_k, min_relevance).

`GET /rag/documents` lista los documentos indexados con sus rutas relativas, cantidad de chunks y fecha de indexación.

`POST /rag/search` permite realizar búsquedas de prueba contra la base de conocimiento con un query y número de resultados configurables, retornando contenido, fuente y relevancia porcentual.

`POST /rag/sync` dispara la sincronización de documentos ejecutando `rag_admin.py sync` como subproceso, y reinicializa la instancia del sistema RAG para cargar los cambios.

Los endpoints principales `/ask` (audio) y `/ask_text` (texto) integran RAG de forma transparente, incluyendo en su respuesta JSON los campos `rag_sources` (lista de fuentes únicas) y `rag_chunks` (fragmentos con contenido, fuente, sección y relevancia).

---

## 7. Integración en el Frontend

### 7.1 Propagación de Fuentes RAG

La interfaz React recibe los `rag_chunks` del backend y los propaga a través de todo el flujo de renderizado de mensajes. La función `addReply(text, modelInfo, ragChunks)` acepta tres parámetros, y los tres call sites del código — `processAudio` (procesamiento de audio), `submitTextPrompt` (envío de texto), y `resendMessage` (reenvío de mensajes) — extraen y propagan `data.rag_chunks` desde la respuesta del servidor.

Cada mensaje del modelo en el estado `conversation` almacena los chunks como propiedad `ragChunks`, lo que permite que las fuentes persistan mientras dure la sesión de chat y se rendericen junto a la respuesta correspondiente.

### 7.2 Panel de Fuentes Colapsable

Cuando una respuesta del modelo incluye fuentes RAG, se renderiza un componente `<details>` colapsable inmediatamente debajo del texto de respuesta. El trigger muestra "📚 Fuentes (N)" indicando la cantidad de fragmentos utilizados. Al expandir, se presenta cada chunk en una tarjeta con fondo oscuro semitransparente (`bg-gray-800/80`) que incluye:

El nombre del archivo fuente con estilo destacado en gris claro, posicionado a la izquierda de la tarjeta. El porcentaje de relevancia en color índigo a la derecha, calculado como `(chunk.relevance * 100).toFixed(0)`. El nombre de la sección en cursiva cuando está disponible. El contenido del fragmento, truncado a 300 caracteres en la interfaz (el servidor ya lo trunca a 500) con indicador de elipsis.

Esta implementación permite que el usuario verifique las fuentes de cada respuesta sin saturar la interfaz: las fuentes permanecen ocultas por defecto y solo se revelan cuando el usuario las solicita explícitamente.

---

## 8. Herramienta de Administración CLI

### 8.1 Comandos Disponibles

El script `rag_admin.py` expone ocho comandos a través de `argparse` con subparsers:

`sync` es el comando principal de operación. Escanea recursivamente el directorio `documents/` buscando archivos con extensiones soportadas, compara hashes MD5 contra el índice existente, procesa archivos nuevos o modificados, elimina chunks de archivos borrados, y persiste el índice actualizado. La salida reporta conteos de archivos agregados, actualizados y eliminados.

`add <archivo>` permite ingestar un archivo específico sin necesidad de ubicarlo en el directorio `documents/`. Valida la extensión, procesa el archivo, indexa los chunks resultantes y actualiza el índice.

`remove <fuente>` busca coincidencias parciales en los nombres de archivos indexados y elimina todos los chunks asociados a cada coincidencia.

`list` muestra todos los documentos indexados con su conteo de chunks y fecha de indexación.

`search <consulta> [-n N]` ejecuta una búsqueda semántica y muestra los N resultados más relevantes (por defecto 3) con su fuente, porcentaje de relevancia y un extracto de contenido.

`stats` presenta estadísticas agregadas: documentos fuente, chunks totales, modelo de embeddings, fecha de última sincronización, tamaño en disco de la base de datos, y distribución por tipo de archivo.

`rebuild [--force]` elimina completamente la base de datos vectorial (incluyendo el directorio `rag_data/`) y ejecuta una sincronización desde cero. Sin el flag `--force`, solicita confirmación antes de proceder.

`check` verifica la disponibilidad de todas las dependencias del sistema: bibliotecas Python (chromadb, python-docx, pandas, openpyxl, pypdf, tabulate, httpx), disponibilidad del servidor Ollama, instalación del modelo de embeddings, y existencia del directorio de documentos.

---

## 9. Gestión del Contexto Conversacional

### 9.1 Ventana de Contexto y Sumarización

El sistema implementa gestión inteligente del contexto conversacional para prevenir la degradación de calidad que ocurre cuando el historial excede la capacidad del modelo. Se configura una ventana de contexto de 8192 tokens (`CONTEXT_WINDOW_SIZE`) y un límite de 20 mensajes (`MAX_HISTORY_MESSAGES`).

Cuando el historial supera el límite, la función `manage_context` divide los mensajes en dos grupos: los 6 más recientes (`KEEP_RECENT_MESSAGES`) se preservan intactos, mientras que los anteriores se condensan en un resumen generado por `qwen2.5:0.5b`. El resumen resultante se inyecta en el system prompt como "Resumen de la conversación anterior", proporcionando contexto histórico sin consumir tokens de ventana de contexto.

### 9.2 Interacción entre Contexto Conversacional y RAG

La construcción del system prompt sigue un orden de prioridad deliberado. El contexto RAG se inyecta primero y se marca como "PRIORIDAD ALTA" porque la información documental verificada debe prevalecer sobre conversaciones previas y contexto general del usuario. El resumen conversacional se ubica después para proporcionar continuidad. Esta jerarquía asegura que, ante tokens limitados de contexto, los documentos RAG ocupen la posición más favorable para influir en la generación del modelo.

---

## 10. Ventajas de la Arquitectura Implementada

### 10.1 Privacidad y Soberanía de Datos

Toda la infraestructura opera localmente. Los documentos corporativos nunca se transmiten a servicios externos. Los embeddings se generan en la máquina local mediante Ollama, la base vectorial reside en el sistema de archivos local, y la inferencia LLM ocurre en hardware propio. Esto cumple potencialmente con requerimientos de protección de datos como GDPR o normativas sectoriales que prohíben el procesamiento de información sensible en servidores de terceros.

### 10.2 Costo Operativo Nulo

Una vez descargados los modelos (una operación que se realiza una sola vez), el sistema no genera costos recurrentes. No hay facturación por token, por request, ni por almacenamiento en la nube. El costo se reduce al hardware local y la electricidad consumida, lo cual representa una ventaja significativa frente a servicios como la API de OpenAI donde cada consulta incrementa la factura.

### 10.3 Actualización Incremental de Conocimiento

Los documentos se pueden agregar, modificar o eliminar en cualquier momento. La sincronización incremental basada en hashes MD5 asegura que solo los archivos afectados se reprocesan, manteniendo tiempos de actualización proporcionales al cambio realizado y no al tamaño total de la base documental.

### 10.4 Trazabilidad y Transparencia

Cada respuesta del modelo puede acompañarse de las fuentes documentales exactas que la sustentan, con indicación del archivo, sección y porcentaje de relevancia. El usuario puede verificar la información expandiendo el panel de fuentes, lo que genera confianza y permite detectar posibles errores del modelo.

### 10.5 Modularidad y Desacoplamiento

Los procesadores de documentos, el motor de embeddings, la base vectorial, el servidor web y la interfaz de usuario operan como componentes independientes con interfaces bien definidas. El procesador de documentos puede extenderse a nuevos formatos agregando una función al diccionario `PROCESSORS`. La función de embeddings puede reemplazarse por otro modelo simplemente cambiando la variable `EMBEDDING_MODEL`. La interfaz de usuario puede consumir los endpoints RAG independientemente de cómo se generen las respuestas.

---

## 11. Limitaciones Identificadas y Mejoras Propuestas

### 11.1 Limitaciones del MVP Actual

**Chunking estático.** La estrategia de chunking actual utiliza un tamaño fijo de 1000 caracteres con solapamiento de 100. Este enfoque no distingue entre tipos de contenido: un párrafo narrativo y una tabla de ingredientes reciben el mismo tratamiento, aunque sus características semánticas difieren significativamente. Un chunk que corta una tabla por la mitad pierde la coherencia de los datos que contiene.

**Ausencia de re-ranking.** Los resultados de la búsqueda vectorial se filtran únicamente por un umbral de distancia coseno. No existe una etapa de re-ranking que utilice un modelo cross-encoder para refinar la relevancia de los candidatos recuperados. Los embeddings biencoder (como `nomic-embed-text`) son eficientes para la recuperación inicial pero menos precisos que un cross-encoder para determinar la relevancia fina entre query y documento.

**Modelo de embeddings monolingüe.** Aunque `nomic-embed-text` funciona razonablemente bien con español, está optimizado principalmente para inglés. Textos con mezcla de idiomas, tecnicismos culinarios o jerga regional pueden no representarse con la misma fidelidad semántica.

**Sin OCR ni procesamiento de imágenes en documentos.** Los archivos PDF que contienen texto como imágenes escaneadas no se procesan correctamente porque `pypdf` solo extrae texto embebido, no realiza reconocimiento óptico de caracteres.

**Ventana de contexto compartida.** El contexto RAG compite por tokens con el historial conversacional y el system prompt. En conversaciones largas con muchos documentos relevantes, los fragmentos RAG podrían truncarse o no entrar completos.

**Single-threaded para embeddings.** La generación de embeddings durante la ingesta se realiza secuencialmente, documento por documento. Para bases documentales grandes, esto puede resultar lento.

### 11.2 Mejoras Propuestas para Iteraciones Futuras

**Chunking semántico adaptativo.** Implementar un chunker que analice la estructura del contenido para decidir los puntos de corte. Las tablas deberían mantenerse como unidades atómicas. Los procedimientos paso a paso deberían conservarse completos. Los párrafos narrativos podrían dividirse en oraciones usando un tokenizador de lenguaje natural como `spaCy`.

**Pipeline de re-ranking con cross-encoder.** Agregar una etapa posterior a la búsqueda vectorial que utilice un modelo cross-encoder (como `cross-encoder/ms-marco-MiniLM-L-6-v2`) para revaluar y reordenar los candidatos. Esto mejoraría significativamente la precisión de los resultados, especialmente para consultas ambiguas.

**Embeddings multilingües.** Reemplazar `nomic-embed-text` por un modelo de embeddings optimizado para español o multilingüe como `multilingual-e5-large` o `paraphrase-multilingual-MiniLM-L12-v2`, que ofrece representaciones semánticas más precisas para textos en español.

**Integración de OCR.** Incorporar `Tesseract` o `EasyOCR` para procesar PDFs escaneados, permitiendo que documentos históricos digitalizados se incorporen a la base de conocimiento.

**Ingesta asíncrona y en paralelo.** Refactorizar la ingesta para generar embeddings en batch asincrónicamente, aprovechando la capacidad de Ollama de procesar múltiples textos en una sola llamada al endpoint `/api/embed` con el parámetro `input` como lista.

**Panel de administración web.** Desarrollar una interfaz web integrada en el frontend para gestionar la base de conocimiento: subir documentos, monitorear el estado de indexación, ejecutar búsquedas de prueba y visualizar estadísticas, eliminando la dependencia de la CLI.

**Soporte para metadatos enriquecidos.** Permitir que los usuarios asignen categorías, etiquetas o niveles de confidencialidad a los documentos, habilitando filtros que restrinjan la búsqueda a subconjuntos específicos de la base de conocimiento.

**Evaluación cuantitativa del RAG.** Implementar un framework de evaluación con datasets de test (pares pregunta-respuesta) que permita medir métricas como precisión de recuperación, recall, y calidad de la respuesta generada, para guiar las decisiones de optimización con datos objetivos.

---

## 12. Requerimientos Mínimos e Instrucciones de Despliegue

### 12.1 Requisitos de Hardware

El sistema requiere una computadora con al menos 8 GB de RAM para ejecutar los modelos más ligeros (gemma3:4b, qwen2.5:0.5b y nomic-embed-text simultáneamente). Para utilizar modelos más grandes como phi4:latest (14B parámetros), se recomiendan 16 GB de RAM o superior. El almacenamiento necesario es de aproximadamente 5 GB para los modelos base y 500 MB adicionales por cada 10,000 chunks indexados en ChromaDB. No se requiere GPU dedicada, aunque su presencia acelera significativamente la inferencia y la generación de embeddings.

### 12.2 Requisitos de Software

El entorno requiere Python 3.10 o superior, Node.js 18 o superior para el frontend React, y Ollama instalado como servicio local. Las dependencias de Python se instalan mediante:

```bash
pip install fastapi uvicorn httpx numpy scipy
pip install openai-whisper
pip install chromadb python-docx pandas openpyxl pypdf tabulate
```

Las dependencias del frontend se instalan mediante:

```bash
cd chatbot-ui
npm install
```

### 12.3 Configuración Inicial de Modelos

Antes de utilizar el sistema, los modelos de Ollama deben descargarse:

```bash
# Modelo de embeddings (obligatorio para RAG)
ollama pull nomic-embed-text

# Modelos de inferencia
ollama pull gemma3:4b        # Chat general
ollama pull phi4:latest      # Razonamiento matemático
ollama pull qwen2.5:0.5b     # Router/clasificador (ultra ligero)

# Modelo de visión (opcional, para análisis de imágenes)
ollama pull llava:7b
```

### 12.4 Preparación de la Base de Conocimiento

Los documentos fuente se colocan en el directorio `documents/`, que soporta subdirectorios organizados según la taxonomía que el usuario considere apropiada. El escaneo recursivo mediante `rglob("*")` explora toda la jerarquía de carpetas automáticamente.

```bash
# Verificar dependencias
python rag_admin.py check

# Sincronizar documentos
python rag_admin.py sync

# Verificar indexación
python rag_admin.py stats

# Probar búsqueda
python rag_admin.py search "ingredientes del pollo juanillo"
```

### 12.5 Inicio de Servicios

El sistema requiere tres procesos ejecutándose simultáneamente:

```bash
# Terminal 1: Servidor Ollama (si no está activo como servicio del sistema)
ollama serve

# Terminal 2: Servidor backend FastAPI
python -m uvicorn gemma_server:app --host 0.0.0.0 --port 8000

# Terminal 3: Frontend React
cd chatbot-ui
npm run dev
```

La interfaz estará disponible en `http://localhost:5173` y se comunicará con el backend en `http://localhost:8000`.

### 12.6 Verificación de Funcionamiento

Para confirmar que el sistema RAG opera correctamente:

```bash
# Verificar el estado del RAG desde la API
curl http://localhost:8000/rag/status

# Probar una búsqueda RAG desde la API
curl -X POST "http://localhost:8000/rag/search?query=pollo+juanillo&n_results=3"

# Verificar la salud general del sistema
curl http://localhost:8000/health
```

Una respuesta correcta del endpoint `/rag/status` mostrará `"enabled": true`, `"initialized": true`, junto con los conteos de documentos y chunks indexados.

---

## 13. Estructura de Archivos del Proyecto

```
proyecto/
├── gemma_server.py              # Servidor FastAPI principal (orquestador)
├── rag_module.py                # Núcleo RAG: embeddings, ChromaDB, búsqueda
├── document_processors.py       # Extracción y chunking de documentos
├── rag_admin.py                 # CLI de administración de la base de conocimiento
├── requirements.txt             # Dependencias Python
├── .gitignore                   # Exclusiones (rag_data/, documents/*, .venv/)
│
├── documents/                   # Directorio de documentos fuente
│   ├── README.md                # Instrucciones de uso
│   └── recetas/                 # Ejemplo: subdirectorio temático
│       └── pollo juanillo/      # Subdirectorio por receta
│           ├── Receta.xlsx
│           ├── Checklist.docx
│           └── Guion.docx
│
├── rag_data/                    # Base de datos vectorial (persistente, no versionada)
│   ├── chroma.sqlite3           # Almacenamiento ChromaDB
│   └── .index.json              # Índice de archivos procesados con hashes
│
└── chatbot-ui/                  # Frontend React
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx              # Componente principal con integración RAG
        ├── App.css
        ├── main.jsx
        └── index.css
```

---

## 14. Consideraciones de Seguridad

### 14.1 Aislamiento de Datos

La base de datos vectorial y los documentos fuente residen exclusivamente en el sistema de archivos local. No se implementa autenticación en los endpoints del servidor porque el despliegue previsto es estrictamente local (`localhost`). Si el sistema se expusiera en una red, sería imperativo agregar autenticación JWT o API keys en los endpoints RAG y restringir los orígenes CORS a dominios específicos.

### 14.2 Sanitización de Contenido

Los textos extraídos de documentos se procesan tal cual, sin sanitización contra inyección de prompts. Un documento maliciosamente construido podría contener instrucciones que el modelo interpretaría como directivas, potencialmente alterando su comportamiento. En un entorno donde los documentos provienen exclusivamente de fuentes confiables internas, este riesgo es aceptable, pero debería abordarse si la plataforma se abre a documentos de terceros.

### 14.3 Control de Acceso a Documentos

Actualmente no existe control de acceso granular a los documentos indexados. Cualquier consulta puede acceder a cualquier chunk de la base de conocimiento. En entornos con documentos de diferentes niveles de confidencialidad, se debería implementar un sistema de roles y filtros de metadata que restrinja qué documentos son accesibles para cada usuario o grupo.

---

## 15. Métricas y Observabilidad

### 15.1 Logging Actual

El sistema implementa logging en consola utilizando `print()` con prefijos semánticos: `📚` para operaciones RAG, `🔀` para enrutamiento, `📝` para sumarización, `✂️` para truncamiento de historial, y `⚠️` para advertencias. Cada request registra la consulta del usuario, el modelo seleccionado, la categoría de enrutamiento, el conteo de documentos RAG relevantes encontrados, y la respuesta generada.

### 15.2 Métricas Recomendadas para Producción

Para monitorear la efectividad del sistema RAG en producción, se deberían implementar las siguientes métricas: tiempo de generación de embeddings para consultas (latencia P50 y P99), tiempo de búsqueda ChromaDB, distribución de relevancia de los resultados recuperados, ratio de consultas sin resultados RAG versus total, tiempo total de respuesta end-to-end incluyendo transcripción Whisper cuando aplica, y uso de memoria por parte de ChromaDB y los modelos cargados.

---

## 16. Conclusiones

El sistema RAG implementado demuestra la viabilidad de construir un asistente conversacional con conocimiento especializado utilizando exclusivamente infraestructura local y modelos de código abierto. El pipeline completo — desde la ingesta de documentos Word y Excel hasta la presentación de respuestas con fuentes citadas en una interfaz web moderna — opera sin dependencias de servicios cloud, garantizando la privacidad de los datos y eliminando costos recurrentes.

La principal complejidad técnica del proyecto no residió en la arquitectura conceptual del RAG, que sigue un patrón bien establecido en la industria, sino en la integración práctica entre las diferentes versiones de las herramientas involucradas. La adaptación de `OllamaEmbeddingFunction` para compatibilizar ChromaDB 1.5 con la API de embeddings de Ollama ilustra un desafío recurrente en sistemas que combinan múltiples bibliotecas de código abierto en rápida evolución: las interfaces cambian entre versiones minor, y la documentación no siempre refleja los cambios oportunamente.

El MVP cumple con los objetivos funcionales establecidos: los documentos se indexan correctamente, las búsquedas semánticas retornan resultados relevantes, las respuestas del modelo priorizan el conocimiento documental sobre la generación especulativa, y la interfaz de usuario presenta las fuentes de forma transparente. Las mejoras identificadas — chunking semántico, re-ranking, embeddings multilingües y panel de administración web — representan optimizaciones incrementales que pueden implementarse sin modificar la arquitectura fundamental.
