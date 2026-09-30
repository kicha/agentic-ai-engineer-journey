from langchain_groq import ChatGroq

from app.config.settings import QueryingConfig
from app.retrieval.hyde.hyde_generator import HyDEGenerator


def main():

    querying_config = QueryingConfig()
    llm = ChatGroq(
        model=querying_config.llm_model,
        temperature=0.0,
    )

    hyde = HyDEGenerator(llm)

    queries = [
        "What does ef_construct control?",
        "How does the M parameter affect HNSW?",
        "What metadata can be stored with vectors?",
        "Why does HNSW avoid comparing every vector?",
    ]

    for query in queries:

        result = hyde.generate(query)

        print("\n" + "=" * 80)
        print("ORIGINAL QUERY")
        print("=" * 80)
        print(result.original_query)

        print("\nHYPOTHETICAL DOCUMENT")
        print("-" * 80)
        print(result.hypothetical_document)


if __name__ == "__main__":
    main()
