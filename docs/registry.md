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

- **It lists a remote server:** the address clients reach over Streamable HTTP, and no package: a client does not install
  a deployment.
- **With the key** (`--auth key`, the default), the entry says a client must send a secret `Authorization` header. The
  key itself is never in the entry.
- **With OAuth** (`--auth oauth`), the entry names no header. The client learns which issuer to ask from the server's
  protected resource metadata ([serving.md](serving.md#with-tokens-from-your-identity-provider-oauth)).
- **The description** is at most 100 characters, the registry's limit. An environment's defaults to its name. A CLM
  scenario has none of its own, so give one.
- **The address must be HTTPS and name a host.** The registry refuses `localhost` and `127.0.0.1`; `clm-registry` also
  refuses every other bare IP address, so an entry names a host other computers can find. An internal name such as
  `decisions.corp.internal` is accepted, and so is a path before `/mcp`, as behind a gateway
  (`https://gateway.corp/tools/decisions/mcp`).
- **Nothing secret goes in the address.** A user, password, query or fragment is refused: an entry is published as
  written and never fully removed, and the key belongs in the header.
- **The repository** must be a GitHub or GitLab project's home page, all the registry takes. Leave it out for any other forge.

Check the file with the registry's own tool, then publish it:

```bash
mcp-publisher validate server.json
mcp-publisher login github --registry https://registry.example.org   # or dns, http, github-oidc
mcp-publisher publish server.json
```

`mcp-publisher` comes from the registry's releases (`brew install mcp-publisher`, or the binaries on GitHub).

## Public or private

**The public registry** (`registry.modelcontextprotocol.io`) is for a deployment meant for the public, such as a
demonstration. It lists only what anyone can reach: a publicly accessible remote address, or a package on a public index.

- A name under `io.github.<you>/` needs a GitHub login.
- A name under a GitHub organisation needs a personal access token with `read:org`, or GitHub Actions, because
  `mcp-publisher login github` cannot grant organisation names yet.
- A name under your own domain needs a DNS or HTTP proof.
- Entries are never fully removed. With `mcp-publisher status`, a version can be marked `deprecated`, which still lists
  it with a warning, or `deleted`, which hides it from listings but can be undone.

**A private registry** suits a deployment on a private network, as most decision servers will be. It is the same software
and API, run by your organisation. Clients inside the network search it, and nothing about the deployment leaves the
network. Who may publish depends on how it is run:

- **GitHub login:** any GitHub user who can reach it may publish under their own `io.github.<user>/` name. It does not
  limit publishing to an organisation's members.
- **Your own identity provider (OIDC):** the claims a token must carry decide who may publish and edit.
- **DNS or HTTP proof:** whoever holds a domain's key may publish under that domain.

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
   host. So listen on `127.0.0.1` only, as above, and never turn it on for a registry others use.
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
   header it needs. With the key in `DECISION_SERVER_KEY`, `uv run clm-call <address>` lists the server's tools, and
   `clm-agent <address> "QUESTION"` asks a model to use them.

In the check, a client found the entry, saw it needed the `Authorization` header, sent the key and called the tools. The
internal host name does not resolve on a laptop, so the client was pointed at the local server for that name.
