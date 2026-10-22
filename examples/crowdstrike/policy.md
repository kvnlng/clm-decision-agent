# This demo's disruption policy

> A made-up policy for the CrowdStrike-day demo. It is not any real airline's policy.

When a flight is cancelled or badly delayed, the passenger writes in. Read their message and their
booking, and give the answer of the first rule below that fits. The sections after the rules say
what each answer means; they are also what the decision server reads as its route descriptions.

## How to decide (the first that fits)

1. The message is too vague to tell what they need: **Agent**.
2. They no longer want to travel: **Refund**.
3. Their own flights still go, arrive within an hour of plan, and any connection still works:
   **No_Action**.
4. One of our own flights today gets the whole party there sooner than their own flights now will,
   and by any time they must arrive by: **Rebook**, on the one that arrives first.
5. Failing that, their fare allows a partner airline, and a partner's flight today does it:
   **Rebook_Partner**, on the one that arrives first.
6. Their own flights still go, nothing is faster, and they arrive by any time they must:
   **No_Action**.
7. One of our own flights tomorrow before noon has seats for the whole party, and they have no time
   they must arrive by that it misses: **Overnight**, on the one that arrives first.
8. They must arrive by a time nothing meets: **Refund** if the fare is refundable, otherwise
   **Agent**.
9. Otherwise: **Agent**.

A time they would like to arrive by, but need not, changes nothing: the flight that arrives first is
also the best for it.

## Rebook

The passenger still wants to travel, and one of our own airline's flights today has seats for the
whole party and gets them there sooner than their own flight now will, in time for anything they
must not miss. Book them on the flight that arrives first.

## Rebook_Partner

The passenger still wants to travel and none of our own flights today will do: none has seats, or
none arrives in time for something they must not miss. A partner airline's flight today will, and
their fare allows partner airlines. Book them on the partner flight that arrives first.

## Overnight

The passenger still wants to travel, but no flight today has seats for the whole party, and nothing
they mention needs them to arrive tonight. Book a hotel for tonight and our first flight tomorrow
morning.

## Refund

The passenger no longer wants to travel, or they must arrive by a time that no flight can meet and
their fare is refundable. Refund the fare.

## Agent

The message is too vague to tell what the passenger needs, or nothing else fits: they must arrive by
a time no flight meets and the fare can't be refunded, or there are no seats today or tomorrow
morning. Hand the case to a person.

## No_Action

The passenger's own flights still go and still get them there soonest: a short delay, or a longer
one with nothing faster, and any connection still works. Nothing needs changing.

## Airlines and partners

A regional airline flying for a larger one counts as that airline. The partners are this demo's
own; no real agreement is meant.

| Airline  | Flights operated by    | Partner          |
|----------|------------------------|------------------|
| Delta    | DL, Endeavor (9E)      | Alaska (AS)      |
| United   | UA                     | JetBlue (B6)     |
| American | AA, PSA (OH), Envoy (MQ) | Alaska (AS)    |

Every other airline, SkyWest (OO) and Republic (YX) included, stands alone with no partner.
