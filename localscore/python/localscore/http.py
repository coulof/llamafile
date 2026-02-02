"""HTTP client for submitting benchmark results to LocalScore API."""

import json
import time
from typing import Optional, Tuple

import requests

BASE_URL = "https://www.localscore.ai"
API_ENDPOINT = "/api/results"


def get_user_confirmation() -> str:
    """Ask user for confirmation to submit results."""
    print("\nDo you want to submit your results to https://localscore.ai? The results will be public (y/n): ", end="")
    try:
        user_input = input().strip().lower()
        return user_input
    except (EOFError, KeyboardInterrupt):
        return "n"


def submit_results(
    payload: dict,
    auto_submit: bool = False,
    skip_submit: bool = False,
    verbose: bool = False,
    max_retries: int = 3,
) -> Tuple[bool, Optional[str]]:
    """
    Submit benchmark results to the LocalScore API.

    Args:
        payload: The JSON payload to submit
        auto_submit: If True, submit without asking for confirmation
        skip_submit: If True, skip submission entirely
        verbose: If True, print detailed information
        max_retries: Maximum number of retry attempts

    Returns:
        Tuple of (success, result_url or error_message)
    """
    if skip_submit:
        print("\nResults Not Submitted.")
        return False, None

    # Ask for confirmation unless auto-submit is enabled
    if not auto_submit:
        confirmation = get_user_confirmation()
        if confirmation not in ("yes", "y"):
            print("\nResults Not Submitted.")
            return False, None

    if verbose:
        print(f"Submitting results...\n Payload: {json.dumps(payload, indent=2)}")
    else:
        print("\nSubmitting results...")

    # Attempt submission with retries
    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 2 ** attempt
            print(f"Retry attempt {attempt + 1} of {max_retries} after {wait_time} seconds...")
            time.sleep(wait_time)

        try:
            response = requests.post(
                f"{BASE_URL}{API_ENDPOINT}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    if "id" in data:
                        result_url = f"{BASE_URL}/result/{data['id']}"
                        print(f"Result Link: {result_url}")
                        return True, result_url
                    else:
                        print("Error: Response missing result ID")
                        continue
                except json.JSONDecodeError:
                    print("Error parsing response JSON")
                    continue
            else:
                print(f"Error submitting results to the public database. Status: {response.status_code}")
                if attempt < max_retries - 1:
                    continue

        except requests.exceptions.Timeout:
            print("Error: Request timed out")
            if attempt < max_retries - 1:
                continue

        except requests.exceptions.ConnectionError as e:
            print(f"Error: Connection failed - {e}")
            if attempt < max_retries - 1:
                continue

        except requests.exceptions.RequestException as e:
            print(f"Error submitting results: {e}")
            if attempt < max_retries - 1:
                continue

    print(f"Failed to submit results after {max_retries} attempts")
    return False, None


def prepare_submission_payload(
    json_data: dict,
    summary: dict,
) -> dict:
    """
    Prepare the final payload for API submission.

    Args:
        json_data: The benchmark results data from JSONPrinter
        summary: The results summary dictionary

    Returns:
        Complete payload ready for submission
    """
    payload = json_data.copy()
    payload["results_summary"] = summary
    return payload
