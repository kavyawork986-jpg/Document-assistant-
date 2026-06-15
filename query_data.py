import argparse
import re
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rag_utils import (
    get_chat_model,
    get_embeddings,
    load_environment,
)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

STOPWORDS = {
    # common English
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "has", "have", "had", "he", "her", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "my", "myself", "no", "not", "of", "off", "on", "or", "our",
    "out", "she", "so", "some", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "too",
    "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "will", "with", "you", "your", "but", "about",
    "also", "just", "did", "do", "does", "would", "could", "should",
    "than", "both", "each", "any", "all", "few", "more", "most",
    "other", "such", "only", "own", "same", "again", "further",
    # company document filler
    "section", "clause", "article", "paragraph", "document", "policy",
    "procedure", "guidelines", "handbook", "manual", "agreement",
    "pursuant", "accordance", "herein", "hereto", "thereof", "thereto",
    "whereby", "whereas", "aforementioned", "abovementioned",
    "shall", "may", "must", "please", "note", "refer", "see",
    "above", "below", "following", "per", "etc", "vs", "re",
    "effective", "dated", "version", "updated", "revised", "page",
    "applicable", "relevant", "respective", "provided", "subject",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\b\w+\b", text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").strip())
    return [part.strip() for part in parts if part.strip()]


def extract_subject_tokens(question: str) -> list[str]:
    words = re.findall(r"\b[A-Za-z][A-Za-z']*\b", question)
    lowered = [word.lower() for word in words]
    return [
        word
        for word, lowered_word in zip(words, lowered, strict=False)
        if lowered_word not in STOPWORDS
    ]


def is_speech_question(question: str) -> bool:
    lowered_question = question.lower()
    return lowered_question.startswith("what did") and any(
        verb in lowered_question
        for verb in (" say", " said", " ask", " asked", " reply", " replied", " cry", " cried", " call", " called")
    )


def rank_sentences(question: str, sentences: list[str]) -> list[tuple[float, str]]:
    question_tokens = tokenize(question)
    candidate_sentences: list[tuple[float, str]] = []
    subject_tokens = [token.lower() for token in extract_subject_tokens(question)]
    lowered_question = question.lower()

    for sentence in sentences:
        sentence_tokens = tokenize(sentence)
        lowered_sentence = sentence.lower()
        if not sentence_tokens:
            continue

        overlap = question_tokens & sentence_tokens
        score = float(len(overlap))

        if question.lower().startswith("who"):
            if any(token in lowered_sentence for token in subject_tokens):
                score += 1.5
            if re.search(r"\b(is|was)\b|'s", lowered_sentence):
                score += 1.0

        if is_speech_question(question) and any(
            token in lowered_sentence for token in subject_tokens
        ):
            if any(
                verb in lowered_sentence
                for verb in ("say", "said", "asked", "cried", "replied", "called")
            ):
                score += 2.0
            if '"' in sentence or "“" in sentence:
                score += 0.5

        if question.lower().startswith("where") and any(
            word in lowered_sentence
            for word in ("in ", "at ", "under", "over", "near", "inside")
        ):
            score += 0.5

        if question.lower().startswith("when") and any(
            char.isdigit() for char in sentence
        ):
            score += 0.5

        if subject_tokens and all(token in lowered_sentence for token in subject_tokens):
            score += 0.5

        if lowered_question in lowered_sentence:
            score += 3.0

        if score > 0:
            candidate_sentences.append((score, sentence))

    candidate_sentences.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidate_sentences


def answer_from_context(question: str, documents: list[Document]) -> str:
    sentence_pool: list[str] = []
    for doc in documents:
        sentence_pool.extend(sentence_split(doc.page_content))

    ranked_sentences = rank_sentences(question, sentence_pool)

    if not ranked_sentences:
        return documents[0].page_content.strip()

    best_sentences: list[str] = []
    for _score, sentence in ranked_sentences:
        if sentence not in best_sentences:
            best_sentences.append(sentence)
        if question.lower().startswith("who") and len(best_sentences) == 1:
            break
        if is_speech_question(question) and len(best_sentences) == 1:
            break
        if len(best_sentences) == 2:
            break

    return " ".join(best_sentences)


def main():
    load_environment()

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved context before the answer.",
    )
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function, provider = get_embeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search(query_text, k=6)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = get_chat_model()
    if model is None:
        response_text = answer_from_context(query_text, results)
    else:
        try:
            response_text = model.invoke(prompt).content
        except Exception:
            response_text = answer_from_context(query_text, results)

    sources = [doc.metadata.get("source", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    if args.show_context:
        formatted_response = (
            f"Retrieved context:\n{context_text}\n\n{formatted_response}"
        )
    print(formatted_response)


if __name__ == "__main__":
    main()
