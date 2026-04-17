class Agent:
    def generate_answer(self, query, ranked_docs):
        context = " ".join([d["document"] for d in ranked_docs])
        return f"Answer based on context: {context} | Question: {query}"