# Handling sensitive text

Read this before real text goes into a decision server built on this framework. It describes what the code does; it
is not legal or compliance advice, and nothing here makes a deployment compliant with any regulation.

## Where text goes

The text a caller sends with `decide`, `decide_many`, `check_call` or `correct` reaches:

| What | Where it is written | Who can read it | How to keep it out |
|---|---|---|---|
| Every call | CLM, in the server's process (`[clm]`, either backend). The first start downloads CLM from Hugging Face; requests never go there | The process | Not possible |
| Calls that run tools | Your informing and acting tools, and whatever they call. Acting tools get the raw text, identifiers included; no language model rewords it. The example's `create_ticket` puts it in the ticket | Wherever tools send it | Your tools module and `[tools.settings]`: have tools return and send the least they can |
| `correct`, in review mode only | The corrections record, `<scenario>.corrections.jsonl` | Owner only (below) | `[record]`; no review mode |
| Corrections you train from | The examples file and the trained heads, unencrypted | Whoever reads those files | See below |
| Only with `--trace FILE --trace-text` | The trace: a scenario's `decide`, `decide_many`, `check_call` and `correct` text; an environment's tool-call arguments. Without `--trace-text` the trace has no text | Owner only, like the record | Leave `--trace-text` off |
| The server's key, on every HTTP call | As a bearer header. `clm-agent` sends it over HTTPS or to this computer only, never over plain http to another host | Key holders: they can call the server and see its decisions and what its informing tools return | Keep the key secret; HTTPS when callers are elsewhere ([serving.md](serving.md)) |

Nothing leaves the machine except what your tools send. Outside review mode and `--trace-text`, the server saves no
request text. The language model in `[generator]` is used only by `clm-synthesize`, which sends it the scenario's
route descriptions and examples, never requests.

## Keep sensitive text out

This is the protection that matters most, and the framework cannot do it for you.

- Design routes and questions so the decision never needs patient-level details. The example decides "what kind of
  request" and "how urgent", not anything about a record.
- Give the scenario a route for requests it must not handle, and make the acting tools skip it.
- Tell the people behind the calling agents what not to type. **Nothing stops them typing it anyway.**

## Protect the record

```toml
[record]
path = "private/my.corrections.jsonl" # somewhere other than beside the scenario
key_file = "record.key"               # encrypt each line with this key ...
# key_env = "RECORD_KEY"              # ... or with a key in an environment variable
```

- The record, and a folder made for it, are created readable by their owner only. In review mode the server also
  makes an older record so.
- `uv run clm-record newkey record.key` makes a key. Keep it out of version control and back it up: a record written
  with a key cannot be read without it.
- `uv run clm-record purge <scenario>` deletes the record after asking, and only a file that is plainly a corrections
  record, since `path` can name any file.

## What encryption does not do

- It protects the record file at rest. Anyone with the key, or with access to the running process, can read the text.
- It is not de-identification. The text is unchanged, and it is still sent to CLM and the tools.
- The examples file and the trained heads are not encrypted. If you train from corrections, the heads are derived from
  that text; store them as carefully as the record.
- There is no key rotation. To change the key, purge the record and start a new one.
- In review mode the server will not start on a record that has lines under another key, since saving more would leave
  a file no single key can read. Without review mode, the server and `clm-train` say how many lines they could not read.
- Lines written before you set a key stay unencrypted in the same file.
- Server addresses in a scenario may not contain a user name or password; put a key in the variable named by
  `api_key_env`, which is never printed.

## Before using real text

- [ ] Check where CLM runs (`[clm]`), and what each of your tools sends, and where.
- [ ] Decide who gets the server's key, and whether the server is reachable from other machines.
- [ ] Decide where the record, the examples file and the trained heads live, who can read them, and for how long.
- [ ] If the record is encrypted, decide who holds the key and where its backup is.
- [ ] Send something the server should route out of scope, and see what its tools do with it.
- [ ] Have whoever is responsible for information governance where you work look at this list.
