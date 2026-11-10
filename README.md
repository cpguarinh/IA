# 🎓 Asistente RAG de Orientación Académica y Permanencia Estudiantil
**ACA 2 · Computación en la Nube** — Especialización en Inteligencia Artificial, CUN
Docente: Dr. Hamilton Rivera Flor · Pilar 2: Educación

🔗 **App en línea:** https://rag-educativo-cun.streamlit.app/

## Integrantes
- Juan Felipe Rodríguez Cataño
- Claudia Patricia Guarín Hernández
- Raúl Albeiro Cuéllar
- Andrés Mauricio Moya Romero
- Juan Sebastián Moya Romero

## Descripción
Chatbot RAG que responde consultas sobre reglamento académico, alertas tempranas y bienestar estudiantil del *Colegio Integrado Horizonte* (institución ficticia), fundamentando cada respuesta en documentos indexados en Pinecone y citando la fuente. Incluye diagnóstico de estudiantes reales del dataset *Students Performance* y un panel de gestión documental y monitoreo vectorial.

## Stack tecnológico
| Componente | Herramienta |
|---|---|
| Base vectorial | Pinecone Serverless (`aca1-educacion-gemini`, 768 dim, coseno) |
| Embeddings | Gemini Embedding 2 (`gemini-embedding-2`) |
| LLM generativo | Google Gemini Flash (Cloud API), temperatura 0.2 |
| Orquestación RAG | Python nativo (`google-genai` + `pinecone`) |
| Interfaz | Streamlit |

## Estructura del repositorio
| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit (chat RAG, gestión documental, monitor vectorial) |
| `requirements.txt` | Dependencias |
| `01_…pdf`, `02_…pdf`, `03_…pdf` | Corpus documental institucional |
| `Student_performance_data.csv` | Dataset de rendimiento estudiantil (Kaggle) |

## Ejecución local
```bash
git clone https://github.com/cpguarinh/IA.git
cd IA
pip install -r requirements.txt
```
Crear un archivo `.env` en la raíz:
```env
PINECONE_API_KEY=pcsk_...
GEMINI_API_KEY=AIza...
```
Ejecutar:
```bash
streamlit run app.py
```
La app abre en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud
1. En share.streamlit.io: **Create app** → repositorio `cpguarinh/IA`, rama `main`, archivo `app.py`.
2. En **Advanced settings → Secrets**:
```toml
PINECONE_API_KEY = "pcsk_..."
GEMINI_API_KEY = "AIza..."
```
3. **Deploy**.

## Seguridad
Las credenciales se gestionan con `.env` (local) o `st.secrets` (nube) y **nunca** se publican en el repositorio; `.gitignore` excluye `.env` y `.streamlit/secrets.toml`.
