"""The portal: one page for the person who administers a decision server, served by the server itself, and a
command-line chat.

    clm-server ENVIRONMENT.toml --http [--port 8000] --portal 8400 [--portal-host 127.0.0.1]
               [--portal-public-origin URL] [--ask-backend openai|claude] [--ask-model M] [--ask-model-url U]
    clm-chat URL [--model M] [--model-url U] [--backend openai|claude] [--dry-run [TOOL]] [--full]

The page shows an environment's whole configuration and changes only its documents and its derived layer, each
change taking effect at once; sign in with the key in CLM_PORTAL_KEY. `clm-chat` talks to any running MCP server
through a model, printing every call and result.
"""
PORTAL_KEY_ENV = "CLM_PORTAL_KEY"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
MIN_KEY = 16        # clm-server sets no minimum for DECISION_SERVER_KEY, so this is the portal's own
