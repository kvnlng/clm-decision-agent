"""Named queries on Neo4j: each query's `cypher` runs with its parameters, and its rows come back as dicts.

The neo4j driver is the optional `neo4j` extra; it is imported only when an
environment's graph backend is neo4j. load_graph.py copies an environment's CSV graph into the database.
"""
from __future__ import annotations

import warnings

from .scenario import ScenarioError, StepError

# Bolt, plain or with TLS; the driver's http:// support is a preview that needs further packages
SCHEMES = ("bolt://", "bolt+s://", "bolt+ssc://", "neo4j://", "neo4j+s://", "neo4j+ssc://")


def _first_line(error: Exception) -> str:
    return (str(error).strip().splitlines() or [type(error).__name__])[0]


def driver_errors() -> tuple[type[Exception], ...]:
    """The driver's own errors: the server's (Neo4jError) and the connection's (DriverError)."""
    from neo4j.exceptions import DriverError, Neo4jError  # pylint: disable=import-outside-toplevel
    return Neo4jError, DriverError


def connect(uri: str, user: str, password: str):
    """A driver that has reached the database; raises ScenarioError in one line when it cannot."""
    try:
        import neo4j  # pylint: disable=import-outside-toplevel
        from neo4j.warnings import PreviewWarning  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ScenarioError("graph.backend: neo4j needs the driver: install the neo4j extra, "
                            "pip install 'clm-decision-agent[neo4j]'") from error
    if not uri.startswith(SCHEMES):
        raise ScenarioError(f"graph.neo4j: cannot connect to {uri}: the uri must start with "
                            f"{' or '.join(SCHEMES)}")
    driver = None
    try:
        # The driver warns about its preview HTTP support on every start, which only clutters a one-line error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PreviewWarning)
            driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except driver_errors() as error:
        if driver is not None:
            driver.close()
        raise ScenarioError(f"graph.neo4j: cannot connect to {uri}: {_first_line(error)}") from error
    return driver


class Neo4jConnector:  # pylint: disable=too-few-public-methods
    """Runs a query's cypher in one database; `errors` are the driver errors to report as the query's failure."""

    def __init__(self, driver, uri: str, database: str, errors: tuple[type[Exception], ...] | None = None):
        self.driver, self.uri, self.database = driver, uri, database
        self.errors = errors if errors is not None else driver_errors()

    def run(self, query, params: dict) -> list[dict]:
        """The query's rows."""
        try:
            with self.driver.session(database=self.database) as session:
                return [record.data() for record in session.run(query.cypher, params)]
        except self.errors as error:
            raise StepError(f"query {query.name}: Neo4j at {self.uri}: {_first_line(error)}") from error
