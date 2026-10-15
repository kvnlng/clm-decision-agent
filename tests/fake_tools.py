"""Tools for the tests: the invoice mocks, and two form-submitting tools (one that always fails)."""
from langchain_core.tools import ToolException, tool

SETTINGS = {}
CONFIGURED = []
CATALOGUE = []      # what list_invoices returns; a test sets it


def configure(settings, scenario_dir):
    """Record what the loader passed, so a test can check it."""
    CONFIGURED.append(str(scenario_dir))
    SETTINGS.clear()
    SETTINGS.update(settings, scenario_dir=str(scenario_dir))


@tool
def lookup_invoice(invoice_number: str) -> dict:
    """Look up an invoice and its charges by invoice number."""
    return {
        "invoice_number": invoice_number,
        "charges": [
            {"amount": 49.99, "date": "2026-09-28"},
            {"amount": 49.99, "date": "2026-09-28"},
        ],
    }


@tool
def issue_refund(invoice_number: str, amount: float) -> dict:
    """Refund a specific charge on an invoice."""
    return {"invoice_number": invoice_number, "refunded": amount, "status": "refund issued"}


@tool
def submit_form(summary: str) -> dict:
    """Submit the completed form."""
    return {"id": "F-1", "summary": summary}


@tool
def broken_submit(summary: str) -> dict:
    """Submit the completed form to a service that is down."""
    raise ToolException("The service could not be reached.")


@tool
def list_invoices() -> list:
    """List the invoices a customer might mean."""
    if CATALOGUE == ["fail"]:
        raise ToolException("The billing system could not be reached.")
    return CATALOGUE


TOOLS = [lookup_invoice, issue_refund, submit_form, broken_submit, list_invoices]
