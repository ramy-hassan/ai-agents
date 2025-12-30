from llm_based_agents.without_lang_chain.llm import call_llm


class Agent:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        # append everything that happen in the ReAct loop
        self.messages = []

    def __call__(self, message: str):
        self.messages.append({"role": "user", "content": message})
        result = self.execute(self.messages)
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self, messages):
        return call_llm(self.system_prompt, messages)
