"""Tools for the tests: plain functions that take the request, as the framework calls them."""
# Tools take the request whether or not they use it
# pylint: disable=unused-argument

SETTINGS = {}
CALLS = []          # (tool name, request) for each call, so a test can check what a tool was given


def configure(settings, scenario_dir):
    """Record what the loader passed, so a test can check it."""
    SETTINGS.clear()
    SETTINGS.update(settings, scenario_dir=str(scenario_dir))


def lookup(request):
    """An informing tool."""
    CALLS.append(("lookup", request))
    return {"facts": ["The queue is A."]}


def act(request):
    """An acting tool."""
    CALLS.append(("act", request))
    return {"id": 1}


def broken(request):
    """A tool whose service is down."""
    raise RuntimeError("The service could not be reached.")


def unserialisable(request):
    """A tool that returns something JSON cannot hold."""
    return {"when": object()}


TOOLS = [lookup, act, broken, unserialisable]
