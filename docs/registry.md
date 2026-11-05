# Listing a deployment in an MCP registry

An MCP registry is a catalogue clients and people search to find MCP servers. Each deployment of the decision server, with
its own scenario or environment, address and key, is a server of its own, so list the deployment, not the framework.

## The entry

`clm-registry` writes a deployment's `server.json`, the format registries take:

```bash
uv run clm-registry examples/tariff/environment.toml --name io.github.you/tariff \
    --url https://decisions.example.org/mcp --version 0.1.0 [--auth key|oauth] \
    [--description "..."] [--repository https://github.com/you/repo]
```

The entry lists a remote server, the address clients reach over Streamable HTTP, and no package: a client does not
install a deployment.

| Field | Takes |
|---|---|
| File | The scenario or `environment.toml` served; read and checked even with `--description`, so a mistyped path is refused |
| `--name` | `namespace/name`, such as `io.github.you/tariff` |
| `--url` | HTTPS, a host name, a path ending in `/mcp`. An internal name (`decisions.corp.internal`) and a path before `/mcp` (`https://gateway.corp/tools/decisions/mcp`, behind a gateway) are fine. Refused: `localhost` and `127.0.0.1` (by the registry) and every other bare IP address (by `clm-registry`), since an entry names a host other computers can find; and any user, password, query or fragment, since an entry is published as written and never fully removed, and the key belongs in the header |
| `--version` | Such as `1.0.0` |
| `--auth key` (default) | Says a client must send a secret `Authorization` header; the key itself is never in the entry |
| `--auth oauth` | Names no header; the client learns which issuer to ask from the server's protected resource metadata ([serving.md](serving.md#with-tokens-from-your-identity-provider-oauth)) |
| `--description` | At most 100 characters, the registry's limit; defaults to an environment's name, and a CLM scenario has none, so give one |
| `--repository` | A GitHub or GitLab project's home page, all the registry takes; leave it out for any other forge |

Check the file with the registry's own tool, then publish it:

```bash
mcp-publisher validate server.json
mcp-publisher login github --registry https://registry.example.org   # or dns, http, github-oidc
mcp-publisher publish server.json
```

`mcp-publisher` comes from the registry's releases (`brew install mcp-publisher`, or the binaries on GitHub).

## Public or private

**The public registry** (`registry.modelcontextprotocol.io`) is for a deployment meant for the public, such as a
demonstration, and lists only what anyone can reach: a public remote address, or a package on a public index. A name
under `io.github.<you>/` needs a GitHub login; one under a GitHub organisation needs a personal access token with
`read:org`, or GitHub Actions, since `mcp-publisher login github` cannot grant organisation names yet; one under your own
domain needs a DNS or HTTP proof. Entries are never fully removed: `mcp-publisher status` can mark a version
`deprecated`, still listed with a warning, or `deleted`, hidden from listings but undoable.

**A private registry** suits a deployment on a private network, as most decision servers will be. It is the same software
and API, run by your organisation; clients inside the network search it, and nothing about the deployment leaves the
network. Who may publish depends on how it is run: with GitHub login, any GitHub user who can reach it, under their own
`io.github.<user>/` name, not only an organisation's members; with your own identity provider (OIDC), whoever holds the
claims a token must carry; with DNS or HTTP proof, whoever holds a domain's key.

## A private registry on this computer

Checked on a Mac with the registry's release 1.8.6 and PostgreSQL 18 from Homebrew, both run natively.

1. Download `registry_darwin_arm64.tar.gz` and `mcp-publisher_darwin_arm64.tar.gz` from
   [the registry's releases](https://github.com/modelcontextprotocol/registry/releases), and check them against the
   release's checksums file.
2. Start PostgreSQL with its own data folder (`initdb`, then `postgres -D <folder> -p 5433`, with `LC_ALL` set to a
   locale such as `en_US.UTF-8`), and create a database `mcp-registry`.
3. Start the registry:

   ```bash
   MCP_REGISTRY_SERVER_ADDRESS=127.0.0.1:8090 \
   MCP_REGISTRY_DATABASE_URL="postgres://mcpregistry@localhost:5433/mcp-registry?sslmode=disable" \
   MCP_REGISTRY_JWT_PRIVATE_KEY=$(openssl rand -hex 32) \
   MCP_REGISTRY_SEED_FROM= MCP_REGISTRY_ENABLE_ANONYMOUS_AUTH=true MCP_REGISTRY_OIDC_ENABLED=false \
   ./registry
   ```

   Anonymous publishing is for trying it out only: anyone who can reach the registry can publish, change or deprecate
   names under `io.modelcontextprotocol.anonymous/`, including an entry that sends clients, and their key, to another
   host. Listen on `127.0.0.1` only, as above, and never turn it on for a registry others use.
4. Write and publish the entry:

   ```bash
   uv run clm-registry examples/tariff/environment.toml \
       --name io.modelcontextprotocol.anonymous/clm-tariff \
       --url https://decisions.corp.internal/mcp --version 0.1.0 \
       --description "Sourcing decisions over a supplier graph, DMN rules and what-if scenarios"
   mcp-publisher login none --registry http://localhost:8090
   mcp-publisher publish server.json
   ```

5. Find it: `curl "http://localhost:8090/v0.1/servers?search=clm-tariff"` gives the entry, with its address and the
   header it needs. The MCP Inspector lists the server's tools with that header
   (`--cli <address> --transport http --header "Authorization: Bearer <the key>" --method tools/list`;
   [serving](serving.md#inspecting-a-server)); with the key in `DECISION_SERVER_KEY`, `clm-agent <address> "QUESTION"`
   asks a model to use them. The internal host name does not resolve on a laptop, so in the check the client was
   pointed at the local server for that name.
