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
- **The address must be HTTPS and name a host.** The registry refuses `localhost` and
  `127.0.0.1`. `registry.py` also refuses every other bare IP address, so that an entry names a
  host other computers can find. An internal name such as `decisions.corp.internal` is
  accepted. A path before `/mcp`, as behind a gateway, is fine
  (`https://gateway.corp/tools/decisions/mcp`).
- **Nothing secret goes in the address.** A user, password, query or fragment is refused: an
  entry is published as written and never fully removed, and the key belongs in the header.
- **The repository** must be a GitHub or GitLab project's home page, which is all the registry
  takes. Leave it out for any other forge.

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
`io.github.<you>/` needs a GitHub login. A name under a GitHub organisation needs a personal
access token with `read:org`, or GitHub Actions, because `mcp-publisher login github` cannot
grant organisation names yet. A name under your own domain needs a DNS or HTTP proof. Entries
are never fully removed. With `mcp-publisher status`, a version can be marked `deprecated`,
which still lists it with a warning, or `deleted`, which hides it from listings but can be undone. Use it for a deployment meant for the public, such as
a demonstration.

**A private registry** suits a deployment on a private network, which is how most decision
servers will run. It is the same registry software, with the same API, run by your
organisation. Clients and agents inside the network search it, and nothing about the deployment
leaves the network. Who may publish is set by how the registry is run. With GitHub login, any
GitHub user who can reach it may publish under their own `io.github.<user>/` name; it does not
limit publishing to an organisation's members. With your own identity provider (OIDC), the
claims a token must carry decide who may publish and edit. With DNS or HTTP proof, whoever holds
a domain's key may publish under that domain.

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
   MCP_REGISTRY_SERVER_ADDRESS=127.0.0.1:8090 \
   MCP_REGISTRY_DATABASE_URL="postgres://mcpregistry@localhost:5433/mcp-registry?sslmode=disable" \
   MCP_REGISTRY_JWT_PRIVATE_KEY=$(openssl rand -hex 32) \
   MCP_REGISTRY_SEED_FROM= MCP_REGISTRY_ENABLE_ANONYMOUS_AUTH=true MCP_REGISTRY_OIDC_ENABLED=false \
   ./registry
   ```

   Anonymous publishing is for trying it out only: it lets anyone who can reach the registry
   publish, change or deprecate names under `io.modelcontextprotocol.anonymous/`. That includes
   an entry that sends clients, and their key, to another host. So listen on `127.0.0.1` only,
   as above, and never turn it on for a registry others use.
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
