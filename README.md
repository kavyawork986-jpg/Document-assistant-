# GitLab Knowledge Base — Question Answering Assistant

Ask questions about GitLab's company handbook in plain English and get answers that come straight from the actual documents — not made up.

You can ask about GitLab's values, mission, communication style, leadership, HR policies, pay, benefits, and more.

This project is built using **RAG (Retrieval-Augmented Generation)**. That's a fancy name for a simple idea: instead of letting an AI guess an answer from memory, we first *find* the most relevant pieces of the documents, then ask the AI to answer *using only those pieces*. This keeps answers accurate and based on real sources.

---

## A few terms explained

If some words below sound technical, here's what they mean in plain language:

| Term | What it actually means |
|------|------------------------|
| **RAG** | A method where the app looks up relevant text first, then asks the AI to answer based on it. |
| **Embedding** | A way of turning words and sentences into numbers so the computer can measure how similar two pieces of text are in meaning. |
| **Vector database** | A special storage place that holds those numbers and can quickly find the text closest in meaning to your question. (This project uses one called **Chroma**.) |
| **Chunk** | A small piece of a document. Big files are cut into smaller chunks so the app can find and use just the relevant parts. |
| **Semantic search** | Searching by *meaning* instead of exact words. Asking "how is pay reviewed?" can still find a section titled "compensation cycles." |
| **LLM** | Large Language Model — the AI (like GPT) that writes the final answer in normal sentences. |
| **Embedding model** | The tool that creates the embeddings. This project uses `all-MiniLM-L6-v2`, which runs on your own computer and needs no API key. |

---

## What's inside the knowledge base

The app already comes loaded with these public GitLab handbook documents:

| Document | What it covers |
|----------|----------------|
| `gitlab_values.md` | CREDIT values — Collaboration, Results, Efficiency, DIB, Iteration, Transparency |
| `gitlab_mission.md` | The company's mission and what GitLab is building |
| `gitlab_communication.md` | How GitLab communicates (mostly written, async-first) |
| `gitlab_leadership.md` | Leadership principles, decision-making, and growth programs |
| `gitlab_anti_harassment.md` | Anti-harassment policy and how to report issues |
| `gitlab_hiring.md` | How GitLab hires people |
| `gitlab_compensation.md` | Salary, equity, RSUs, and pay transparency |
| `gitlab_benefits.md` | Medical, parental leave, pension, and other perks |
| `gitlab_people_group.md` | HR teams, contacts, and the employee journey |

Want to add your own documents? Just drop any `.md` file into the `data/` folder and rebuild the database (explained in the steps below).

---

## How it works (the short version)

1. Every document in the `data/` folder is split into small **chunks**.
2. Each chunk is turned into numbers (**embeddings**) and saved in the **Chroma** database.
3. When you ask a question, the app finds the chunks closest in meaning to your question.
4. Those chunks are handed to the **AI**, which writes a clear answer based only on them.

So you always get answers grounded in the real handbook.

---

## Project structure

```
RAG project/
├── data/                            # The source documents (add new .md files here)
│   ├── gitlab_values.md
│   ├── gitlab_mission.md
│   ├── gitlab_communication.md
│   ├── gitlab_leadership.md
│   ├── gitlab_anti_harassment.md
│   ├── gitlab_hiring.md
│   ├── gitlab_compensation.md
│   ├── gitlab_benefits.md
│   └── gitlab_people_group.md
├── chroma/                          # The vector database (created automatically)
│   └── embedding_config.json
├── create_database.py               # Splits and stores all documents from data/
├── query_data.py                    # Answers your questions
├── compare_embeddings.py            # Compares how similar two words are
├── rag_utils.py                     # Shared helpers (embeddings, AI, config)
├── pyrightconfig.json               # Editor config for the virtual environment
├── requirements.txt                 # The list of packages to install
└── .env                             # Your secret API keys (never share this file)
```

---

## What you need before starting

- **Python 3.10 or newer** installed on your computer
- An **OpenRouter API key** — this lets the app talk to the AI that writes the answers. You can get one for free at [openrouter.ai](https://openrouter.ai).

---

# How to run it on your PC — step by step

Follow these steps in order. Every command is typed into your terminal (Command Prompt on Windows, or Terminal on Mac/Linux).

### Step 1 — Open the project folder

Go into the project folder in your terminal:

```bash
cd "RAG project"
```

### Step 2 — Create a virtual environment

A *virtual environment* is a private, isolated space for this project's packages, so they don't clash with other Python projects on your computer.

```bash
python -m venv venv
```

### Step 3 — Turn the virtual environment on

**On Mac or Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

You'll know it worked when you see `(venv)` appear at the start of your terminal line.

### Step 4 — Install the required packages

This reads the `requirements.txt` file and installs everything the project needs:

```bash
pip install -r requirements.txt
```

(This may take a couple of minutes the first time.)

### Step 5 — Add your API key

Create a file named `.env` in the project folder and put your key inside it:

```
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_token_here        # optional — only removes a HuggingFace warning message
```

If you want to use a different AI model, you can add this line too (it defaults to `openai/gpt-4o-mini` if you leave it out):

```
OPENROUTER_MODEL=openai/gpt-4o-mini
```


### Step 6 — Build the database

This reads all the documents, chops them into chunks, and saves them. You only need to do this **once** (or again whenever you add or change documents).

```bash
python create_database.py
```

You should see something like:

```
Split 9 documents into 54 chunks.
Saved 54 chunks to chroma/ using huggingface embeddings.
```

### Step 7 — Ask a question!

Now you're ready. Type your question in quotes:

```bash
python query_data.py "What are GitLab's core values?"
```

Some more examples you can try:

```bash
python query_data.py "What is GitLab's anti-harassment policy?"
python query_data.py "How does GitLab handle compensation reviews?"
python query_data.py "What is GitLab's mission?"
python query_data.py "How does GitLab approach communication?"
```

### Optional — See the source text behind an answer

If you also want to see the exact chunks the answer was built from, add `--show-context`:

```bash
python query_data.py "What are GitLab's core values?" --show-context
```

### Optional — Compare two words

Curious how the app measures meaning? This small tool shows how similar two words are:

```bash
python compare_embeddings.py
```

---

That's it — you now have a working question-answering assistant running on your own computer. 
