# Import depdendencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP("Shopping Assistant MCP local server")

@mcp.tool("get_offers_by_retailer")
async def get_offers_by_retailer(retailer: str) -> str:
    """Return offers available on a spcific retialer"""
    return ("retailer " + retailer + " has 20% off on all products")


@mcp.tool("get_nearest_store")
async def get_nearest_store(location: str) -> str:
    """Return the nearest store to the given location."""
    return ("The nearest store is at 123 Main St")


@mcp.tool("get_offers_for_product")
async def get_offers_for_product(product: str) -> str:
    """Return the current offers for the specified product."""
    return ("Offer of " + product + " is 20% off")


@mcp.tool("add_to_shopping_cart")
async def add_to_shopping_cart(product: str) -> str:
    """Add the specified product to the shopping cart."""
    return ("Added" + product + "to cart")

if __name__ == "__main__":
    mcp.run(transport="stdio")