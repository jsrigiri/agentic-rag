class Agent:
    def generate_answer(self, query, context_docs):
        context = " ".join(context_docs)
        return f"Answer based on context: {context} | Question: {query}"