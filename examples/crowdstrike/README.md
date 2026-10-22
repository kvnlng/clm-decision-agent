# The CrowdStrike day

On 19 July 2024 a faulty CrowdStrike update crashed Windows machines around the world. US
airlines cancelled 3,548 flights that day, Delta alone 1,325, and hundreds of thousands of
passengers needed a new plan at once. This example replays that day: a decision server has to
decide, for every passenger who writes in, how to help them.

The flights are real. Everything else (seats, passengers, fares, messages and the rebooking
policy) is made up for the demo, generated from a seed so every run is the same.

## The flights

`flights.csv.gz` holds every US flight of 19 July 2024, and those of 20 July leaving before noon,
with times as minutes from 19 July 00:00 UTC.

Source: Bureau of Transportation Statistics, "Reporting Carrier On-Time Performance (1987-present)",
July 2024. A U.S. Government Work, in the public domain.

To rebuild it, download the July 2024 file from
<https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr> (the
"Prezipped File" for 2024, July), unzip it, and run:

```bash
uv run python examples/crowdstrike/make_extract.py <the unzipped CSV>
```

BTS gives local times without time zones. The script works out each airport's offset from the
schedule itself, and drops the 8 flights whose own schedule contradicts itself.
