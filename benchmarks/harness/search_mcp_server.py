#!/usr/bin/env python3
"""Keyless MCP stdio server exposing a single `web_search` tool backed by
DuckDuckGo's lite HTML endpoint (html.duckduckgo.com returned a bot-check
"anomaly" page from this VPS's IP; lite.duckduckgo.com/lite/ does not and is
used instead — see run-log.md for that finding).

Every query issued through this tool is appended as a JSON line to the file
named by the MOE_BENCH_QUERY_LOG env var (query, results incl. url+snippet,
timestamp), satisfying BENCHMARK.md's requirement to log every query/URL/
snippet for arms A1/A3.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

from mcp.server import MCPServer

mcp = MCPServer("moe-bench-search")

QUERY_LOG = os.environ.get("MOE_BENCH_QUERY_LOG")


def _log(entry: dict) -> None:
    if not QUERY_LOG:
        return
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(QUERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _clean(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).replace("&amp;", "&").replace("&#x27;", "'").strip()


def _unwrap_ddg_redirect(href: str) -> str:
    # lite.duckduckgo.com links go through //duckduckgo.com/l/?uddg=<urlencoded target>
    m = re.search(r"uddg=([^&]+)", href)
    return urllib.parse.unquote(m.group(1)) if m else href


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web (DuckDuckGo lite, keyless) and return top results with URL + snippet."""
    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (moe-benchmark-harness)"})
    results = []
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        links = re.findall(r"<a rel=\"nofollow\" href=\"([^\"]+)\" class='result-link'>(.*?)</a>", html, re.S)
        snippets = re.findall(r"<td class='result-snippet'>(.*?)</td>", html, re.S)
        for i, (href, title) in enumerate(links[:5]):
            results.append({
                "url": _unwrap_ddg_redirect(href),
                "title": _clean(title),
                "snippet": _clean(snippets[i]) if i < len(snippets) else "",
            })
    except Exception as e:
        _log({"query": query, "error": str(e), "results": []})
        return f"search error: {e}"

    _log({"query": query, "results": results})
    if not results:
        return "no results"
    return "\n\n".join(f"{r['title']}\n{r['url']}\n{r['snippet']}" for r in results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
