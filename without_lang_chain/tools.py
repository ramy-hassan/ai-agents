def get_nearest_store(location: str) -> str:
    return ("The nearest store is at 123 Main St")

def get_offers_for_product(product: str) -> str:
    return ("Offer of" + product + "is 20% off")

def add_to_shopping_cart(product: str) -> str:
    return ("Added" + product + "to cart")

known_actions = {
    "get_nearest_store": get_nearest_store,
    "get_offers_for_product": get_offers_for_product,
    "add_to_shopping_cart": add_to_shopping_cart
}
