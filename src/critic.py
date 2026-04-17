class Critic:
    def review(self, answer, task_type="classification"):
        if task_type == "classification":
            return "Answer OK" if len(answer) >= 20 else "Answer too short"

        if task_type == "regression":
            score = min(1.0, max(0.0, len(answer) / 200.0))
            return {"confidence_score": score}

        raise ValueError(f"Unsupported task_type: {task_type}")