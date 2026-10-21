# Handling sensitive text

Read this before real text goes into a decision server built on this framework. It describes
what the code does. It is not legal or compliance advice, and nothing here makes a deployment
compliant with any regulation.

## Where text goes

The text a caller sends with `decide`, `check_call` or `correct` reaches:

| Place | When | You control it with |
|---|---|---|
| CLM | Every call | `[clm]`: it runs in the server's process, with either backend |
| Your informing and acting tools, and whatever they call | When the decision runs them | Your tools module and `[tools.settings]` |
| The corrections record, on disk | `correct`, in review mode only | `[record]` |
| The examples file and trained heads, on disk | If you train from corrections | Where you keep those files |

CLM runs inside the server's process, so nothing leaves the machine except what your tools
send. The first start downloads CLM from Hugging Face; requests are never sent there. The server keeps no log of the requests it is sent: outside review
mode, it saves nothing.

**Acting tools receive the caller's raw text.** There is no language model to reword it, so
whatever the requester typed, identifiers included, can end up wherever a tool sends it. The
example's `create_ticket` puts the text in the ticket.

**Anyone with the key can call an HTTP server**, and see its decisions and the facts its
informing tools return. Keep the key secret, and serve over HTTPS when callers are elsewhere:
[serving.md](serving.md).

The language model named in `[generator]` is used only by `synthesize.py`, which sends it the
scenario's route descriptions and examples, not requests.

## Keep sensitive text out

This is the protection that matters most, and the framework cannot do it for you.

- Design routes and questions so the decision never needs patient-level details. The
  example decides "what kind of request" and "how urgent", not anything about a record.
- Give the scenario a route for requests it must not handle, and make the acting tools skip it.
- Tell the people behind the calling agents what not to type. **Nothing stops them typing it
  anyway.**
- Have tools return and send the least they can.

## Protect the record

The record (`<scenario>.corrections.jsonl`) holds the text of every request the owner
corrected.

```toml
[record]
path = "private/my.corrections.jsonl" # keep it somewhere other than beside the scenario
key_file = "record.key"               # encrypt each line with this key ...
# key_env = "RECORD_KEY"              # ... or with a key held in an environment variable
```

- The file is created readable by its owner only.
- `uv run python record.py newkey record.key` makes a key. Keep it out of version control,
  and back it up: a record written with a key cannot be read without it.
- `uv run python record.py purge <scenario>` deletes the record, after asking.
- `purge` only deletes a file that is plainly a corrections record, since `path` can name any
  file.

## What encryption does not do

- It protects the record file at rest. Anyone with the key, or with access to the running
  process, can read the text.
- It is not de-identification. The text is unchanged, and it is still sent to CLM and the tools.
- The examples file and the trained heads are not encrypted. If you train from corrections, the
  heads are derived from that text; store them as carefully as the record.
- There is no key rotation. To change the key, purge the record and start a new one.
- Lines written before you set a key stay unencrypted in the same file.
- Server addresses in a scenario may not contain a user name or password; put a key in the
  variable named by `api_key_env`, which is never printed.

## Before using real text

- [ ] Check where CLM runs (`[clm]`), and what each of your tools sends, and where.
- [ ] Decide who gets the server's key, and whether the server is reachable from other
      machines.
- [ ] Decide where the record, the examples file and the trained heads live, who can read
      them, and how long they are kept.
- [ ] If the record is encrypted, decide who holds the key and where its backup is.
- [ ] Send something the server should route out of scope, and see what its tools do with it.
- [ ] Have whoever is responsible for information governance where you work look at this
      list.
