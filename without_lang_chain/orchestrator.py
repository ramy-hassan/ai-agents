import regex

from llm_based_agents.without_lang_chain.reAct_agent import Agent
from llm_based_agents.without_lang_chain.tools import known_actions

system_prompt = """You are a shopping assistant that support users in different ways.
You operate in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, you output the final Answer.

Use Thought to explain your reasoning based on the user's request.
Use Action to invoke one of the tools available to you. Always follow the format exactly, and return PAUSE afterward.
Observation will be the result of executing the Action.

Your available actions are:

get_nearest_store:
e.g. get_nearest_store: user_location
Finds the store nearest to the given location.

get_offers_for_product:
e.g. get_offers_for_product: "toothpaste"
Returns a list of current offers for the specified product.

add_to_shopping_cart:
e.g. add_to_shopping_cart: "toothpaste"
Adds the specified product or offer to the user's shopping cart.

Example session:
Question: Where can I find the nearest store?
Thought: I need to look up the store closest to the user's location.
Action: get_nearest_store: Berlin
PAUSE

You will be called again with this:

Observation: The nearest store is "SuperMart Berlin, Alexanderplatz".

You then output:

Answer: The nearest store is "SuperMart Berlin, Alexanderplatz".

---

Question: Are there any deals on toothpaste?
Thought: I should check what current offers are available for toothpaste.
Action: get_offers_for_product: "toothpaste"
PAUSE

You will be called again with this:

Observation: ["Colgate 2-pack - 1.99€", "Oral-B Whitening - 2.49€"]

You then output:

Answer: There are two current offers for toothpaste: "Colgate 2-pack - 1.99€" and "Oral-B Whitening - 2.49€".
""".strip()


def perform(question, max_iterations=5):
    action_regex = regex.compile("^Action: (\w+): (.*)$")
    counter = 0
    bot = Agent(system_prompt)
    next_prompt = question
    while counter < max_iterations:
        counter += 1
        result = bot(next_prompt)
        print(result)
        actions = [action_regex.match(a)
                   for a in result.split("\n")
                   if action_regex.match(a)]

        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise ValueError(f"Unknown action: {action}")
            print("-- Running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("-- Observation: {}".format(observation))
            next_prompt = f"Observation: {observation}"
        else:
            return


question = "Add the best offers of butter around me in Lichtenberg to my shopping cart."

print(perform(question, 5))
