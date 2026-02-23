cd "D:\chat window"
.venv\Scripts\Activate.ps1
# Edit vizzy_chat/.env — add your OPENAI_API_KEY
cd vizzy_chat
streamlit run app.py# 🟣 Vizzy Chat

**A premium AI-powered conversational creative platform.**

Vizzy Chat is a single conversational interface where users can create, transform, iterate, and deploy visual, narrative, and experiential content — powered by OpenAI GPT-4o and DALL-E 3, orchestrated by LangChain + LangGraph.

Built for [Deckoviz](https://deckoviz.com/).

---

## ✨ Features

| Capability                 | Description                                                         |
| -------------------------- | ------------------------------------------------------------------- |
| **Intent Detection**       | LLM-powered classification into 8 creative intents                  |
| **Dual Mode**              | Home (personal creativity) & Business (marketing/brand)             |
| **Multi-Variation Output** | 3 visually distinct variations per request                          |
| **Iteration Engine**       | Natural language refinement ("make it warmer", "more minimal")      |
| **Taste Memory**           | Learns user preferences over time, influences outputs automatically |
| **Multi-Step Creative**    | Story → Scene breakdown → Per-scene image generation                |
| **Marketing Intelligence** | Brand-aware, premium aesthetics, strategic positioning              |
| **Download Support**       | One-click download for every generated image                        |
| **Human-Centred UX**       | Warm status messages, no technical jargon                           |

---

## 🏗️ Architecture

```
vizzy_chat/
│
├── app.py                      # Streamlit UI — chat interface
├── config.py                   # Central configuration (env, constants)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (not committed)
│
├── core/                       # Brain — orchestration & intelligence
│   ├── intent_engine.py        # LangChain LLM intent classification
│   ├── pathway_selector.py     # LangGraph state machine — routes intent → pipeline
│   ├── generation_engine.py    # Executes image, text, multi-step pipelines
│   ├── iteration_engine.py     # Handles refinement & delta changes
│   └── memory_engine.py        # Persistent taste memory (JSON-backed)
│
├── services/                   # API layer — thin wrappers
│   ├── openai_service.py       # OpenAI SDK wrapper (chat, image, embeddings)
│   ├── image_service.py        # Image variation generation, download, conversion
│   └── text_service.py         # Narrative, marketing copy, scene extraction
│
├── storage/
│   └── memory.json             # Persistent user preference store
│
└── utils/
    ├── prompt_builder.py       # Structured prompt templates & builders
    └── logger.py               # Loguru-based logging
```

### Data Flow

```
User Message
    │
    ▼
┌─────────────────┐
│  Intent Engine   │  ← LangChain ChatOpenAI (JSON mode)
│  (classify)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Pathway Selector │  ← LangGraph StateGraph
│ (route)          │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬───────────┬──────────┐
    ▼         ▼          ▼           ▼          ▼
  Image    Text     Multi-Step   Iteration  Conversation
  Pipeline Pipeline  Pipeline    Engine     Pipeline
    │         │          │           │          │
    └────┬────┴──────────┴───────────┴──────────┘
         │
         ▼
┌─────────────────┐
│  Memory Engine   │  ← learns preferences from interaction
│  (learn & save)  │
└────────┬────────┘
         │
         ▼
   Streamlit UI Response
```

---

## 🚀 Setup & Run

### Prerequisites

- Python 3.10+
- OpenAI API key with GPT-4o and DALL-E 3 access

### 1. Clone & Enter

```bash
cd "D:\chat window"
```

### 2. Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r vizzy_chat/requirements.txt
```

### 4. Configure API Key

Edit `vizzy_chat/.env`:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 5. Run

```bash
cd vizzy_chat
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🎯 Design Decisions

### Why LangChain + LangGraph?

- **LangChain** provides a clean abstraction over OpenAI's API with structured output support, retry logic, and model-agnostic interfaces
- **LangGraph** enables the creative pipeline as a **state machine** — each intent routes to a different execution path through conditional edges, making the orchestration explicit and extensible

### Why JSON for Memory?

The MVP uses JSON file storage for zero-dependency persistence. The `memory_engine.py` is designed with a clean interface (`get_preferences()`, `update_preferences()`) so the backend can be swapped to SQLite, Redis, or PostgreSQL without changing any consumer code.

### Why 3 Variations?

Creative work benefits from divergent options. Three variations (bold/dramatic, soft/intimate, abstract/conceptual) give users meaningful choice without overwhelming them. Each variation uses a different aesthetic angle defined in the prompt builder.

### Prompt Engineering Strategy

- **Separation of concerns**: System prompt (role + mode context) is separate from user prompt (request + variation guidance + memory injection)
- **Anti-generic**: Prompts explicitly instruct against cliché AI visuals, stock-photo aesthetics
- **Mode-aware**: Home prompts emphasise emotion and personal resonance; Business prompts emphasise brand strategy and premium perception
- **Memory injection**: User taste preferences are subtly woven into prompts without explicitly mentioning them

---

## 🛣️ Future Scalability Roadmap

| Phase    | Feature                                                   | Effort |
| -------- | --------------------------------------------------------- | ------ |
| **v1.1** | Image upload + transformation (vision API)                | Medium |
| **v1.2** | SQLite/PostgreSQL memory backend                          | Low    |
| **v1.3** | User authentication & multi-user support                  | Medium |
| **v1.4** | Gallery view — browse & re-use past creations             | Low    |
| **v2.0** | Brand profile system (upload logo, colours, guidelines)   | High   |
| **v2.1** | Batch generation (e.g. "10 seasonal visuals")             | Medium |
| **v2.2** | Video/animation loop generation (Runway/Pika integration) | High   |
| **v2.3** | Team collaboration & shared workspaces                    | High   |
| **v3.0** | Embedding-based taste similarity (find similar styles)    | Medium |
| **v3.1** | Plugin system for custom creative pipelines               | High   |
| **v3.2** | API layer for headless/programmatic access                | Medium |

---

## 📝 License

Built for Deckoviz internship assessment. All rights reserved.
