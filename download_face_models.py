"""Download and verify the official OpenCV Zoo YuNet and SFace models."""

from face_models import verify_model_files


def main() -> int:
    print("Downloading or verifying the OpenCV YuNet and SFace face models...")
    try:
        for path in verify_model_files():
            print(f"Verified {path.name}")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Could not prepare the face models: {error}")
        return 1
    print(
        "OpenCV face models are ready. YuNet is MIT-licensed and SFace is Apache-2.0 "
        "under their respective OpenCV Zoo model directories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
