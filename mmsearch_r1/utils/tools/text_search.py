from mmsearch_r1.utils.tools.offline_search import format_text_results, get_offline_index
from mmsearch_r1.utils.tools.serpapi_backend import has_serpapi, search_google


def call_text_search(text_query: str):
    if has_serpapi():
        try:
            return search_google(text_query, num_results=3)
        except Exception as exc:
            print(f"[Warning] SerpApi text search failed, falling back to offline/fake search: {exc}")

    index = get_offline_index()
    if index is not None:
        results = index.search_text(text_query, topk=3)
        tool_returned_str = format_text_results(results)
        tool_stat = {
            "success": True,
            "num_results": len(results),
            "backend": "offline_fvqa",
        }
        return tool_returned_str, tool_stat

    print(
        "[Warning] You are currently using a *fake* implementation of the text search tool.\n"
        "This is a placeholder for demonstration purposes only. The actual search logic is not included due to privacy, licensing, or infrastructure constraints.\n"
        "Set MMSEARCH_OFFLINE_PARQUET to a veRL parquet file to enable offline FVQA retrieval,\n"
        "or replace this function with your own implementation that connects to a real search backend or API."
    )

    tool_stat = {
        "success": True,
        "num_results": 3,
        "backend": "fake",
    }
    tool_returned_str = (
        "[Text Search Results] Below are the text summaries of the most relevant webpages related to your query, ranked in descending order of relevance:\n"
        "1. (webpage link) Summary of webpage content...\n"
        "2. (webpage link) Summary of webpage content...\n"
        "3. (webpage link) Summary of webpage content...\n"
    )
    return tool_returned_str, tool_stat
