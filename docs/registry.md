# Listing a deployment in an MCP registry

An MCP registry is a catalogue of MCP servers that clients and people search to find one, and
then connect to it. The decision server is a framework: each deployment, with its own scenario or
environment, address and key, is a server of its own. So it is each deployment that is listed,
not the framework.

## The entry

`registry.py` writes a deployment's `server.json`, the format registries take:

```bash
uv run python registry.py examples/tariff/environment.toml --name io.github.you/tariff \
    --url https://decisions.example.org/mcp --version 0.1.0 [--auth key|oauth] \
    [--description "..."] [--repository https://github.com/you/repo]
```

- **It lists a remote server:** the address clients reach over Streamable HTTP. It lists no
  package, because a deployment is not something a client installs.
- **With the key** (`--auth key`, the default), the entry says a client must send an
  `Authorization` header and that it is secret. The key itself is never in the entry.
- **With OAuth** (`--auth oauth`), the entry names no header. The client learns which issuer to
  get a token from through the server's protected resource metadata
  ([serving.md](serving.md#with-tokens-from-your-identity-provider-oauth)).
- **The description** is at most 100 characters, which is the registry's limit. For an
  environment it defaults to the environment's name; a CLM scenario has no description of its
  own, so give one.
- **The address must be HTTPS and name a host.** The registry refuses `localhost` and bare IP
  addresses, so `registry.py` refuses them too. An internal name such as
  `decisions.corp.internal` is accepted.

Check the file with the registry's own tool, then publish it:

```bash
mcp-publisher validate server.json
mcp-publisher login github --registry https://registry.example.org   # or dns, http, github-oidc
mcp-publisher publish server.json
```

`mcp-publisher` comes from the registry's releases (`brew install mcp-publisher`, or the
binaries on GitHub).

## Public or private

**The public registry** (`registry.modelcontextprotocol.io`) lists only what anyone can reach:
a remote address that is publicly accessible, or a package on a public index. A name under
`io.github.<you>/` needs a GitHub login. A name under your own domain needs a DNS or HTTP proof.
Entries cannot be removed. A version can only be marked `deprecated` or `deleted`, which hides
it from listings (`mcp-publisher status`). Use it for a deployment meant for the public, such as
a demonstration.

**A private registry** suits a deployment on a private network, which is how most decision
servers will run. It is the same registry software, with the same API, run by your
organisation. Clients and agents inside the network search it, and nothing about the deployment
leaves the network. Who may publish is set by how the registry is run, for example a GitHub
organisation, or your identity provider for administrators.

## A private registry on this computer

This was checked on a Mac with the registry's release 1.8.6 and PostgreSQL 18 from Homebrew, both
run natively.

1. Download `registry_darwin_arm64.tar.gz` and `mcp-publisher_darwin_arm64.tar.gz` from
   [the registry's releases](https://github.com/modelcontextprotocol/registry/releases), and
   check them against the release's checksums file.
2. Start PostgreSQL with its own data folder (`initdb`, then `postgres -D <folder> -p 5433`, with
   `LC_ALL` set to a locale such as `en_US.UTF-8`), and create a database `mcp-registry`.
3. Start the registry:

   ```bash
   MCP_REGISTRY_SERVER_ADDRESS=:8090 \
   MCP_REGISTRY_DATABASE_URL="postgres://mcpregistry@localhost:5433/mcp-registry?sslmode=disable" \
   MCP_REGISTRY_JWT_PRIVATE_KEY=$(openssl rand -hex 32) \
   MCP_REGISTRY_SEED_FROM= MCP_REGISTRY_ENABLE_ANONYMOUS_AUTH=true MCP_REGISTRY_OIDC_ENABLED=false \
   ./registry
   ```

   Anonymous publishing is for trying it out only. It lets anyone publish names under
   `io.modelcontextprotocol.anonymous/`.
4. Write and publish the entry:

   ```bash
   uv run python registry.py examples/tariff/environment.toml \
       --name io.modelcontextprotocol.anonymous/clm-tariff \
       --url https://decisions.corp.internal/mcp --version 0.1.0 \
       --description "Sourcing decisions over a supplier graph, DMN rules and what-if scenarios"
   mcp-publisher login none --registry http://localhost:8090
   mcp-publisher publish server.json
   ```

5. Find it: `curl "http://localhost:8090/v0.1/servers?search=clm-tariff"` gives the entry, with
   its address and the header it needs.

In the check, a client found the entry, saw that it needed the `Authorization` header, sent the
key, and called the server's tools. The entry's internal host name does not resolve on a laptop,
so the client was pointed at the local server for that one name.
