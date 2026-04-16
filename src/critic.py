class Critic:
    def review(self, answer):
        if len(answer) < 20:
            return "Answer too short"
        return "Answer OK"