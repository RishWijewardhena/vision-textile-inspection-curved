# --------------------------
# 12) Download the calibration app
# --------------------------

#!/bin/bash

REPO="RishWijewardhena/ChArUco-Calibration"
ASSET_NAME="ChArUco_Calibration_Linux"
API_URL="https://api.github.com/repos/$REPO/releases/latest"

cd "$PROJECT_DIR" || exit 1


echo "Fetching latest release info..."

ASSET_URL=$(curl -s "$API_URL" | grep browser_download_url | grep "$ASSET_NAME" | cut -d '"' -f 4 | head -n 1)

if [ -z "$ASSET_URL" ]; then
  echo "❌ No Linux asset found in latest release."
  exit 1
fi

FILE_NAME=$(basename "$ASSET_URL")

echo "Downloading $FILE_NAME..."
curl -L -o "$FILE_NAME" "$ASSET_URL"

chmod +x "$FILE_NAME"
echo "✅ Download complete. Run with: ./$FILE_NAME"
