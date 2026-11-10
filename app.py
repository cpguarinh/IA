import os
import io
import re
import time
import requests
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types

st.set_page_config(
    page_title="Asistente RAG Educativo · CUN ACA 2",
    page_icon="🎓",
    layout="wide"
)

load_dotenv(".env", override=True)

def obtener_credencial(nombre):
    if nombre in os.environ and os.environ[nombre]:
        return os.environ[nombre].strip()
    try:
        if nombre in st.secrets:
            return str(st.secrets[nombre]).strip()
    except Exception:
        pass
    return ""

PINECONE_API_KEY = obtener_credencial("PINECONE_API_KEY")
GEMINI_API_KEY   = obtener_credencial("GEMINI_API_KEY")

INDEX_NAME  = "aca1-educacion-gemini"
DIMENSION   = 768
EMBED_MODEL = "gemini-embedding-2"
BASE_URL    = "https://raw.githubusercontent.com/cpguarinh/IA/main"

PDFS = {
    "DOC-01": ("Manual Estudiantil y Reglamento Académico", "01_Manual_Estudiantil_Reglamento_Academico.pdf"),
    "DOC-02": ("Protocolo de Acompañamiento Psicopedagógico y Alertas Tempranas", "02_Protocolo_Acompanamiento_Psicopedagogico_Alertas_Tempranas.pdf"),
    "DOC-03": ("Guía de Bienestar y Actividades Extracurriculares", "03_Guia_Bienestar_Actividades_Extracurriculares.pdf"),
}

SYSTEM_PROMPT = """Eres el Asistente Académico y de Permanencia Estudiantil del Colegio Integrado Horizonte.
Responde ÚNICAMENTE con base en el 'CONTEXTO DOCUMENTAL RECUPERADO' (DOC-01, DOC-02, DOC-03 y DATASET).
Reglas obligatorias:
1. Cita siempre la fuente entre corchetes al respaldar cada afirmación, ej: [DOC-01, Sección 5] o [DATASET, Nivel F].
2. No inventes artículos, umbrales ni reglas que no estén en los fragmentos recuperados.
3. Si el contexto no contiene la respuesta, indica expresamente: 'No cuento con evidencia en los documentos normativos ni en el dataset institucional del Colegio Integrado Horizonte para responder a esta consulta.'"""

@st.cache_data(ttl=3600)
def listar_modelos_gemini(gem_key):
    """Consulta en vivo los modelos Gemini Flash disponibles (los más recientes primero)."""
    try:
        cliente = genai.Client(api_key=gem_key)
        excluir = ("embedding", "image", "tts", "audio", "live", "vision", "thinking")
        nombres = [
            m.name.replace("models/", "") for m in cliente.models.list()
            if "generateContent" in (m.supported_actions or []) and not any(x in m.name for x in excluir)
        ]
        flash = sorted([n for n in nombres if "flash" in n], reverse=True)
        estables = [n for n in flash if "preview" not in n and "exp" not in n and "lite" not in n]
        return estables + [n for n in flash if n not in estables] or sorted(nombres, reverse=True)
    except Exception:
        return []

with st.sidebar:
    st.title("⚙️ Configuración RAG")
    st.caption("Especialización en IA · CUN · Computación en la Nube")

    if not PINECONE_API_KEY:
        PINECONE_API_KEY = st.text_input("🔑 Pinecone API Key", type="password")
    else:
        st.success("Pinecone API Key cargada (.env)")

    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.text_input("🔑 Gemini API Key", type="password")
    else:
        st.success("Gemini API Key cargada (.env)")

    st.divider()
    st.subheader("Parámetros del Modelo")
    modelos_llm = listar_modelos_gemini(GEMINI_API_KEY) if GEMINI_API_KEY else []
    if not modelos_llm:
        st.error("No se encontraron modelos Gemini disponibles para esta API key.")
        st.stop()
    llm_model = st.selectbox("Modelo Generativo (Cloud API)", modelos_llm, index=0)
    top_k = st.slider("Fragmentos a recuperar (Top-k)", min_value=2, max_value=8, value=4)
    temperature = st.slider("Temperatura del LLM", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    filtro_doc = st.selectbox(
        "Filtrar fuente en el Chat",
        ["Todas las fuentes", "DOC-01 (Manual)", "DOC-02 (Psicopedagógico)", "DOC-03 (Bienestar)", "DATASET (Estadísticas)"]
    )

    if st.button("🗑️ Limpiar historial de chat", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Integrantes:**\n"
        "- Juan Felipe Rodríguez Cataño\n"
        "- Claudia Patricia Guarín Hernández\n"
        "- Raúl Albeiro Cuéllar\n"
        "- Andrés Mauricio Moya Romero\n"
        "- Juan Sebastián Moya Romero"
    )

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    st.warning("Configura PINECONE_API_KEY y GEMINI_API_KEY en tu archivo .env o ingrésalas en la barra lateral para continuar.")
    st.stop()

@st.cache_resource
def iniciar_clientes(pc_key, gem_key):
    pc = Pinecone(api_key=pc_key)
    idx = pc.Index(INDEX_NAME)
    gclient = genai.Client(api_key=gem_key)
    return pc, idx, gclient

@st.cache_data(ttl=600)
def cargar_dataset():
    url = f"{BASE_URL}/Student_performance_data.csv"
    if os.path.exists("Student_performance_data.csv"):
        df = pd.read_csv("Student_performance_data.csv")
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
    letras = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}
    df["Nivel"] = df["GradeClass"].astype(int).map(letras)
    return df

@st.cache_data(ttl=600)
def cargar_pdf_secciones(archivo):
    url = f"{BASE_URL}/{archivo}"
    r = requests.get(url, timeout=30)
    reader = PdfReader(io.BytesIO(r.content))
    ruido = re.compile(r"^(Colegio Integrado Horizonte · DOC-|Institución ficticia · uso académico|Página \d+$)")
    sec_re = re.compile(r"^(\d+)\.\s+([A-ZÁÉÍÓÚÑ¿].{2,80})$")
    secciones, actual = [], None
    for num_pag, page in enumerate(reader.pages, start=1):
        for linea in (page.extract_text() or "").splitlines():
            linea = linea.strip()
            if not linea or ruido.match(linea):
                continue
            if sec_re.match(linea):
                actual = {"seccion": linea, "pagina": num_pag, "variables": "General", "texto": []}
                secciones.append(actual)
            elif actual is not None:
                if linea.startswith("Variables relacionadas:"):
                    actual["variables"] = linea.split(":", 1)[1].strip()
                else:
                    actual["texto"].append(linea)
    for s in secciones:
        s["texto"] = " ".join(s["texto"])
    return len(reader.pages), secciones

pc, index, gclient = iniciar_clientes(PINECONE_API_KEY, GEMINI_API_KEY)

def buscar_pinecone(pregunta, k=4, filtro=None):
    t0 = time.time()
    entrada = f"task: search result | query: {pregunta}"
    contenido = [types.Content(parts=[types.Part.from_text(text=entrada)])]
    emb = gclient.models.embed_content(
        model=EMBED_MODEL,
        contents=contenido,
        config=types.EmbedContentConfig(output_dimensionality=DIMENSION)
    )
    vec = list(emb.embeddings[0].values)
    res = index.query(vector=vec, top_k=k, include_metadata=True, filter=filtro)
    lat_ms = (time.time() - t0) * 1000
    return res["matches"], lat_ms

def generar_respuesta_rag(pregunta, matches, historial, modelo, temp):
    bloques = []
    for i, m in enumerate(matches, start=1):
        md = m["metadata"]
        bloques.append(
            f"Fuente #{i}: [{md.get('doc_id', 'N/A')} | {md.get('seccion', 'N/A')} | "
            f"Página {md.get('pagina', 0)} | Similitud: {m['score']:.3f}]\n"
            f"Contenido: {md.get('texto', '')}"
        )
    contexto_str = "\n\n".join(bloques)

    hist_str = ""
    if historial:
        ultimos = historial[-4:]
        hist_str = "HISTORIAL RECIENTE:\n" + "\n".join(
            [f"{h['role'].upper()}: {h['content']}" for h in ultimos]
        ) + "\n\n"

    prompt = (
        f"{hist_str}"
        f"CONTEXTO DOCUMENTAL RECUPERADO:\n{contexto_str}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}"
    )

    t0 = time.time()
    resp = gclient.models.generate_content(
        model=modelo,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temp,
            top_p=0.85,
            max_output_tokens=1024,
        )
    )
    return resp.text, (time.time() - t0) * 1000

st.title("🎓 Sistema RAG de Orientación Académica y Permanencia Estudiantil")
st.markdown("**Colegio Integrado Horizonte** · Base Vectorial Serverless (**Pinecone**) + **Google Gemini API** + **Streamlit**")

tab_chat, tab_docs, tab_vector = st.tabs([
    "💬 Chatbot Conversacional RAG",
    "📂 Gestión Documental y Dataset",
    "🧭 Monitor e Inspector Vectorial (Pinecone)"
])

with tab_chat:
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    with st.expander("🧑‍🎓 Diagnosticar un estudiante real del Dataset (Students Performance)", expanded=False):
        df_est = cargar_dataset()
        col1, col2 = st.columns([1, 3])
        with col1:
            nivel_sel = st.selectbox("Filtrar por Nivel", ["F", "D", "C", "B", "A"], index=0)
            ids_disp = df_est[df_est["Nivel"] == nivel_sel]["StudentID"].tolist()
            id_sel = st.selectbox("Seleccionar StudentID", ids_disp[:50])
        with col2:
            fila = df_est[df_est["StudentID"] == id_sel].iloc[0]
            apoyo_str = ["ninguno", "bajo", "moderado", "alto", "muy alto"][int(fila["ParentalSupport"])]
            pregunta_sugerida = (
                f"Estudiante ID {int(fila['StudentID'])} con GPA {fila['GPA']:.2f} (nivel {fila['Nivel']}), "
                f"{int(fila['Absences'])} inasistencias en el año, {fila['StudyTimeWeekly']:.1f} horas de estudio semanal, "
                f"{'con' if fila['Tutoring'] else 'sin'} tutoría y apoyo familiar {apoyo_str}. "
                f"¿En qué nivel de alerta temprana se encuentra y qué acciones normativas debe aplicar la institución?"
            )
            st.info(pregunta_sugerida)
            if st.button("🚀 Enviar caso al Chatbot RAG"):
                st.session_state.pregunta_rapida = pregunta_sugerida

    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "fuentes" in msg and msg["fuentes"]:
                with st.expander(f"📚 Ver {len(msg['fuentes'])} fragmentos recuperados de Pinecone ({msg['metricas']})"):
                    for f in msg["fuentes"]:
                        md = f["metadata"]
                        st.markdown(
                            f"**[{md.get('doc_id')}] {md.get('seccion')}** · "
                            f"Página: `{md.get('pagina', 0)}` · Similitud coseno: `{f['score']:.4f}`"
                        )
                        st.caption(md.get("texto", ""))
                        st.divider()

    pregunta_usuario = st.chat_input("Escribe tu consulta sobre el reglamento, alertas tempranas, bienestar o rendimiento...")
    if "pregunta_rapida" in st.session_state and st.session_state.pregunta_rapida:
        pregunta_usuario = st.session_state.pop("pregunta_rapida")

    if pregunta_usuario:
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        mapa_filtro = {
            "DOC-01 (Manual)": {"doc_id": {"$eq": "DOC-01"}},
            "DOC-02 (Psicopedagógico)": {"doc_id": {"$eq": "DOC-02"}},
            "DOC-03 (Bienestar)": {"doc_id": {"$eq": "DOC-03"}},
            "DATASET (Estadísticas)": {"doc_id": {"$eq": "DATASET"}},
        }
        filtro_activo = mapa_filtro.get(filtro_doc, None)

        with st.chat_message("assistant"):
            with st.spinner("Consultando Pinecone y generando respuesta fundamentada con Gemini..."):
                matches, lat_pc = buscar_pinecone(pregunta_usuario, k=top_k, filtro=filtro_activo)
                respuesta_txt, lat_llm = generar_respuesta_rag(
                    pregunta_usuario, matches, st.session_state.mensajes, llm_model, temperature
                )
                metricas_str = f"Pinecone: {lat_pc:.0f} ms | Gemini LLM: {lat_llm:.0f} ms"
                st.markdown(respuesta_txt)
                with st.expander(f"📚 Ver {len(matches)} fragmentos recuperados de Pinecone ({metricas_str})"):
                    for f in matches:
                        md = f["metadata"]
                        st.markdown(
                            f"**[{md.get('doc_id')}] {md.get('seccion')}** · "
                            f"Página: `{md.get('pagina', 0)}` · Similitud coseno: `{f['score']:.4f}`"
                        )
                        st.caption(md.get("texto", ""))
                        st.divider()

        st.session_state.mensajes.append({"role": "user", "content": pregunta_usuario})
        st.session_state.mensajes.append({
            "role": "assistant",
            "content": respuesta_txt,
            "fuentes": matches,
            "metricas": metricas_str
        })

with tab_docs:
    st.subheader("📄 Explorador del Corpus Documental (GitHub `cpguarinh/IA`)")
    doc_elegido = st.selectbox(
        "Selecciona un documento institucional para inspeccionar sus secciones:",
        list(PDFS.keys()),
        format_func=lambda k: f"{k} — {PDFS[k][0]}"
    )
    titulo_doc, archivo_doc = PDFS[doc_elegido]
    num_pags, secciones_doc = cargar_pdf_secciones(archivo_doc)

    c1, c2, c3 = st.columns(3)
    c1.metric("Identificador", doc_elegido)
    c2.metric("Páginas del PDF", num_pags)
    c3.metric("Secciones extraídas", len(secciones_doc))

    df_sec = pd.DataFrame([
        {"Sección": s["seccion"], "Página": s["pagina"], "Variables Relacionadas": s["variables"], "Caracteres": len(s["texto"])}
        for s in secciones_doc
    ])
    st.dataframe(df_sec, use_container_width=True, hide_index=True)

    sec_ver = st.selectbox("Leer contenido íntegro de la sección:", [s["seccion"] for s in secciones_doc])
    texto_sec = next(s["texto"] for s in secciones_doc if s["seccion"] == sec_ver)
    st.text_area("Contenido extraído", texto_sec, height=140)

    st.divider()
    st.subheader("📊 Resumen del Dataset Estructurado (`Student_performance_data.csv`)")
    df_completo = cargar_dataset()
    resumen_perfil = (
        df_completo.groupby("Nivel")
        .agg(
            Estudiantes=("StudentID", "count"),
            GPA_Promedio=("GPA", "mean"),
            Inasistencias_Prom=("Absences", "mean"),
            Horas_Estudio_Sem=("StudyTimeWeekly", "mean"),
            Porcentaje_Tutoria=("Tutoring", "mean"),
            Apoyo_Parental=("ParentalSupport", "mean")
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(resumen_perfil, use_container_width=True, hide_index=True)

with tab_vector:
    st.subheader("🧭 Estado del Clúster Serverless en Pinecone")
    stats = index.describe_index_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nombre del Índice", INDEX_NAME)
    m2.metric("Total de Vectores", stats.get("total_vector_count", 0))
    m3.metric("Dimensión Vectorial", stats.get("dimension", DIMENSION))
    m4.metric("Métrica de Similitud", "Coseno (Cosine)")

    st.divider()
    st.subheader("🔎 Probador de Recuperación Semántica y Filtros de Metadatos")
    col_q, col_f1, col_f2 = st.columns([2, 1, 1])
    with col_q:
        query_inspector = st.text_input("Consulta de prueba para el índice:", "¿Cuántas faltas activan el semáforo de alerta temprana?")
    with col_f1:
        filtro_tipo = st.selectbox("Tipo de registro", ["Todos", "documento", "perfil_dataset"])
    with col_f2:
        filtro_var = st.selectbox("Variable asociada", ["Todas", "Absences", "GPA", "Tutoring", "ParentalSupport", "StudyTimeWeekly", "Extracurricular"])

    if st.button("Consultar Índice Vectorial"):
        condiciones = []
        if filtro_tipo != "Todos":
            condiciones.append({"tipo": {"$eq": filtro_tipo}})
        if filtro_var != "Todas":
            condiciones.append({"variables": {"$in": [filtro_var]}})
        filtro_compuesto = {"$and": condiciones} if len(condiciones) > 1 else (condiciones[0] if condiciones else None)

        resultados, lat_insp = buscar_pinecone(query_inspector, k=top_k, filtro=filtro_compuesto)
        st.success(f"Búsqueda completada en {lat_insp:.0f} ms · {len(resultados)} fragmentos recuperados.")
        filas_res = []
        for pos, m in enumerate(resultados, start=1):
            md = m["metadata"]
            filas_res.append({
                "Posición": f"#{pos}",
                "ID Vector": m["id"],
                "Similitud Coseno": round(m["score"], 4),
                "Documento": md.get("doc_id"),
                "Sección": md.get("seccion"),
                "Variables": ", ".join(md.get("variables", [])) if isinstance(md.get("variables"), list) else str(md.get("variables")),
                "Fragmento": md.get("texto", "")
            })
        st.dataframe(pd.DataFrame(filas_res), use_container_width=True, hide_index=True)
