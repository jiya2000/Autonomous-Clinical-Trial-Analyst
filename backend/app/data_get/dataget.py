import json
import requests


BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_diabetes_trials(max_trials: int = 200, page_size: int = 100):
    """Fetch a small set of diabetes-related trials from the v2 API.

    Uses the modern ClinicalTrials.gov REST API:
    https://clinicaltrials.gov/api/v2/studies
    """

    trials = []
    page_token = None

    while len(trials) < max_trials:
        params = {
            "query.term": "type 2 diabetes",
            "pageSize": page_size,
        }

        if page_token:
            params["pageToken"] = page_token

        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        page_trials = data.get("studies", [])
        if not page_trials:
            break

        trials.extend(page_trials)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    # Trim to the requested maximum
    return trials[:max_trials]


def main():
    trials = fetch_diabetes_trials(max_trials=200, page_size=100)

    out = {"studies": trials}

    with open("diabetes_trials.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved", len(trials), "trials to diabetes_trials.json")


if __name__ == "__main__":
    main()