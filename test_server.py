import argparse
import requests
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Test MTailor model deployment")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image file (e.g., n01440764_tench.jpeg)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Cerebrium\localhost endpoint URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API key for Cerebrium",
    )
    parser.add_argument(
        "--custom_test",
        action="store_true",
        help="Run additional platform-level tests (latency, error handling)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image file not found at {args.image_path}")
        return

    if not args.endpoint:
        print("Error: Cerebrium endpoint URL must be provided --endpoint")
        return

    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    with open(args.image_path, "rb") as f:
        files = {"file": f}
        print("Sending image to:", args.endpoint)

        response = requests.post(
            f"{args.endpoint}/predict", files=files, headers=headers
        )

        if response.status_code == 200:
            class_id = response.json().get("class_id")
            print(f"Prediction: class_id = {class_id}")
        else:
            print("Request failed")
            print("Status code:", response.status_code)
            print("Response:", response.text)
            return

    if args.custom_test:
        try:
            resp = requests.get(f"{args.endpoint}/health", headers=headers)
            if resp.status_code == 200:
                print("health endpoint is accessible")
            else:
                print("health endpoint not available")
        except Exception as e:
            print("Failed to access health endpoint:", str(e))


if __name__ == "__main__":
    main()
