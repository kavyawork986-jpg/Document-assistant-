from langchain_classic.evaluation import EvaluatorType, load_evaluator
from langchain_classic.evaluation.schema import PairwiseStringEvaluator
from rag_utils import get_embeddings, load_environment

load_environment()

def main():
    embedding_function, provider = get_embeddings()
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")
    print(f"Embedding provider: {provider}")
    evaluator = load_evaluator(
        EvaluatorType.PAIRWISE_EMBEDDING_DISTANCE,
        embeddings=embedding_function,
    )
    assert isinstance(evaluator, PairwiseStringEvaluator)
    words = ("apple", "iphone")
    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=words[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")

if __name__ == "__main__":
    main()
