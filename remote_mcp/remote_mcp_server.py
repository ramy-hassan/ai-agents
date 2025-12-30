# Import depdendencies
from mcp.server.fastmcp import FastMCP

# Server created
mcp = FastMCP("Shopping Assistant MCP remote server")



@mcp.tool("search_for_offers_by_keyword")
async def search_for_offers_by_keyword(keyword: str) -> str:
    """Return offers that match specific keyword."""
    return ("No offers found for the keyword: " + keyword)


if __name__ == "__main__":
    mcp.run(transport="sse")