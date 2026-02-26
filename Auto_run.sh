#!/bin/bash
set -euo pipefail

# --------------------------
# auto_run.sh installer (FIXED VERSION)
# --------------------------

# Determine target user FIRST (before anything else)
if [ "$EUID" -ne 0 ]; then
  echo "ERROR: Please run as root (sudo). Some steps require sudo."
  echo "Usage: sudo bash $0"
  exit 1
fi

TARGET_USER="${SUDO_USER:-$USER}"
HOME_DIR=$(getent passwd "$TARGET_USER" | cut -d: -f6)

# Validate that we have a valid home directory
if [ -z "$HOME_DIR" ] || [ ! -d "$HOME_DIR" ]; then
  echo "ERROR: Could not determine home directory for user $TARGET_USER"
  exit 1
fi

PROJECT_DIR="$HOME_DIR/Desktop/THREAD"
VENV_DIR="$PROJECT_DIR/venv"
AUTO_RUNNER="$PROJECT_DIR/auto_runner.sh"
SERVICE_PATH="/etc/systemd/system/Thread.service"
REPO_URL="https://github.com/RishWijewardhena/vision-textile-inspection-curved.git"
SOURCE_ENV="$(dirname "$(readlink -f "$0")")/.env"
BRANCH="main"

echo "=========================================="
echo "Thread Installer"
echo "=========================================="
echo "Running as: $(whoami)"
echo "Target user: $TARGET_USER"
echo "Home directory: $HOME_DIR"
echo "Project directory: $PROJECT_DIR"
echo "=========================================="
echo

# --------------------------
# 1) Install system packages
# --------------------------
echo "==> Installing required system packages..."
apt update
apt install -y python3-venv python3-pip acpid git curl

# --------------------------
# 2) Add user to dialout group
# --------------------------
echo
echo "==> Adding user '$TARGET_USER' to group 'dialout' for serial/USB access..."
usermod -a -G dialout "$TARGET_USER" || true
echo "Note: User must log out and log back in (or reboot) for group change to take effect."

# --------------------------
# 3) Create project directory
# --------------------------
echo
echo "==> Setting up project directory: $PROJECT_DIR"
mkdir -p "$PROJECT_DIR"

# --------------------------
# 4) Clone or update GitHub repo
# --------------------------
echo
echo "==> Cloning/updating GitHub repository..."

if [ -d "$PROJECT_DIR/.git" ]; then
    # Case 1: Git repo exists - update it here you will loose your  local updates if it's there
    echo "Repository exists — updating..."
    chown -R "$TARGET_USER":"$TARGET_USER" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    sudo -u "$TARGET_USER" git fetch origin "$BRANCH"
    sudo -u "$TARGET_USER" git reset --hard "origin/$BRANCH"
    
else
    # Case 2: Not a git repo or doesn't exist
    if [ -d "$PROJECT_DIR" ]; then
        echo "Removing existing non-git directory..."
        rm -rf "${PROJECT_DIR:?}"
    fi
    
    echo "Cloning repository..."
    sudo -u "$TARGET_USER" git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
fi

# Ensure correct ownership
chown -R "$TARGET_USER":"$TARGET_USER" "$PROJECT_DIR"

echo "Repository ready at: $PROJECT_DIR"
# Validate that essential files exist
if [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "❌ ERROR: Repository clone/update failed - main.py not found"
    echo "Please check your internet connection and try again"
    exit 1
fi



# --------------------------
# 5) Copy .env file
# --------------------------
echo
echo "==> Copying .env file (if available)..."

ENV_FILE="$PROJECT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  if [ -f "$SOURCE_ENV" ]; then
    cp "$SOURCE_ENV" "$ENV_FILE"
    echo "Copied .env to project directory: $ENV_FILE"
    chown "$TARGET_USER":"$TARGET_USER" "$ENV_FILE"
  else
    echo "❌ ERROR: No .env file found at $SOURCE_ENV"
    echo "The .env file should come with the .sh file"
    exit 1
  fi
else
  echo ".env already exists. Skipping copy."
fi

# Ensure correct ownership
chown -R "$TARGET_USER":"$TARGET_USER" "$PROJECT_DIR"


# --------------------------
# 6) Stop and disable old service if it exists
# --------------------------
echo
echo "==> Checking for old python-script.service..."

OLD_SERVICE="/etc/systemd/system/python-script.service"

if [ -f "$OLD_SERVICE" ]; then
  echo "Found old service. Stopping and disabling..."
  systemctl stop python-script.service || true
  systemctl disable python-script.service || true
  echo "Old service stopped and disabled."
  
else
  echo "No old python-script.service found. Skipping."
fi


# --------------------------
# 7) Create virtualenv & install requirements
# --------------------------
echo
echo "==> Creating virtual environment and installing requirements..."

if [ ! -d "$VENV_DIR" ]; then
  sudo -u "$TARGET_USER" python3 -m venv "$VENV_DIR"
  echo "Virtual environment created at $VENV_DIR"
else
  echo "Virtual environment already exists at $VENV_DIR"
fi

if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "Installing pip packages from requirements.txt..."
  sudo -u "$TARGET_USER" "$VENV_DIR/bin/pip" install --upgrade pip
  sudo -u "$TARGET_USER" "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
  echo "Dependencies installed into venv."
else
  echo "WARNING: No requirements.txt found — skipping pip install."
fi


# --------------------------
# 8) Configure ACPI power button
# --------------------------
echo
echo "==> Configuring ACPI power button to shut down the system..."
tee /etc/acpi/events/powerbtn > /dev/null <<'ACPI_RULE'
event=button/power
action=/usr/sbin/poweroff
ACPI_RULE

systemctl restart acpid
systemctl enable acpid
echo "ACPI configured and acpid restarted."

# --------------------------
# 9) Create auto_runner.sh
# --------------------------
echo
echo "==> Creating helper runner script: $AUTO_RUNNER"

tee "$AUTO_RUNNER" > /dev/null <<EOF
#!/bin/bash
set -euo pipefail

PROJECT_DIR="$PROJECT_DIR"
VENV_DIR="$VENV_DIR"
ENV_FILE="\$PROJECT_DIR/.env"
BRANCH="$BRANCH"

LOG_DIR="\$PROJECT_DIR/logs"
# Daily log file
TODAY=\$(date +%Y-%m-%d)
LOG_FILE="\$LOG_DIR/auto_runner_\$TODAY.log"

mkdir -p "\$LOG_DIR"
touch "\$LOG_FILE"

# Redirect all output to both console and the daily log file
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "======================================="
echo "Auto runner started at \$(date)"
echo "======================================="

cd "\$PROJECT_DIR" || exit 1

# Load .env if present
if [ -f "\$ENV_FILE" ]; then
  set -a
  source "\$ENV_FILE"
  set +a
fi

# Safe default (prevents unbound variable error)
RECEIVE_UPDATES="\${RECEIVE_UPDATES:-true}"

echo "Git remote URL: \$(git config --get remote.origin.url || echo 'unknown')"
echo "Current branch: \$(git branch --show-current || echo 'unknown')"
echo "Current commit: \$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Current commit message: \$(git log -1 --pretty=%B 2>/dev/null || echo 'unknown')"

if [ "\$RECEIVE_UPDATES" = "true" ]; then
  echo "Updates enabled. Checking for updates on \$BRANCH..."

  # Fetch with error handling
  if ! git fetch origin "\$BRANCH" 2>/dev/null; then
    echo "Failed to fetch updates. Continuing with current version."
  else
    echo "Fetch successful."

    # Ensure branch exists
    if ! git show-ref --verify --quiet "refs/heads/\$BRANCH"; then
      echo "Branch \$BRANCH does not exist locally. Creating from origin."
      git checkout -b "\$BRANCH" "origin/\$BRANCH"
    else
      git checkout "\$BRANCH"
    fi

    LOCAL_HASH=\$(git rev-parse "\$BRANCH")
    REMOTE_HASH=\$(git rev-parse "origin/\$BRANCH")

    echo "Local hash: \$LOCAL_HASH"
    echo "Remote hash: \$REMOTE_HASH"

    if [ "\$LOCAL_HASH" != "\$REMOTE_HASH" ]; then
      echo "New updates found."

      # Safe default
      CHANGED_FILES=""

      # Check what changed BEFORE pulling
      CHANGED_FILES=\$(git diff --name-only "\$LOCAL_HASH" "\$REMOTE_HASH" || true)
      echo "Changed files:"
      echo "\$CHANGED_FILES"

      # Pull updates
      if git pull --ff-only origin "\$BRANCH"; then
        echo "Pull successful."
      else
        echo "Fast-forward pull failed. Resetting hard to origin/\$BRANCH"
        git reset --hard "origin/\$BRANCH"
        echo "Reset complete."
      fi

      echo "New commit: \$(git rev-parse HEAD)"
      echo "New commit message: \$(git log -1 --pretty=%B)"

      # Check if requirements.txt changed
      if echo "\$CHANGED_FILES" | grep -q '^requirements.txt\$'; then
        echo "requirements.txt changed. Installing dependencies..."
        "\$VENV_DIR/bin/pip" install --upgrade pip
        "\$VENV_DIR/bin/pip" install -r "\$PROJECT_DIR/requirements.txt"
      else
        echo "No dependency changes."
      fi
    else
      echo "No updates available."
    fi
  fi
else
  echo "Updates disabled. Skipping update check."
fi

# Run the app
if [ -x "\$VENV_DIR/bin/python" ]; then
  exec "\$VENV_DIR/bin/python" "\$PROJECT_DIR/main.py"
else
  exec /usr/bin/python3 "\$PROJECT_DIR/main.py"
fi

EOF
	

chown "$TARGET_USER":"$TARGET_USER" "$AUTO_RUNNER"
chmod +x "$AUTO_RUNNER"
echo "auto_runner.sh created and set executable."


# --------------------------
# 10) Create systemd service file
# --------------------------
echo
echo "==> Creating systemd service: $SERVICE_PATH"

tee "$SERVICE_PATH" > /dev/null <<SERVICE_UNIT
[Unit]
Description=Run Thread main script at boot
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$TARGET_USER
Group=$TARGET_USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$AUTO_RUNNER
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Environment variables (uncomment if GUI needed)
# Environment=DISPLAY=:0
# Environment=XAUTHORITY=$HOME_DIR/.Xauthority

[Install]
WantedBy=multi-user.target
SERVICE_UNIT

chmod 644 "$SERVICE_PATH"
echo "Service file written."

# --------------------------
# 11) Reload systemd and enable service
# --------------------------
echo
echo "==> Reloading systemd and enabling Thread.service..."
systemctl daemon-reload
systemctl enable Thread.service

# --------------------------
# 12) Force Xorg (disable Wayland for AnyDesk compatibility)
# --------------------------
echo
echo "==> Disabling Wayland to force Xorg (for AnyDesk)..."

if [ -f /etc/gdm3/custom.conf ]; then
    sed -i 's/^#WaylandEnable=false/WaylandEnable=false/' /etc/gdm3/custom.conf
    sed -i 's/^WaylandEnable=true/WaylandEnable=false/' /etc/gdm3/custom.conf
    echo "Wayland disabled in /etc/gdm3/custom.conf"
else
    echo "WARNING: /etc/gdm3/custom.conf not found. Skipping Wayland disable."
fi

echo "A reboot is required for this change to take effect."


# --------------------------
# 13) Download the calibration app
# --------------------------


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



echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Review/modify the .env file at: $ENV_FILE"
echo "2. Start the service: sudo systemctl start Thread.service"
echo "3. Check status: sudo systemctl status Thread.service"
echo "4. View logs: sudo journalctl -u Thread.service -f"
echo
echo "Notes:"
echo " - Reboot or log out/in for dialout permissions to apply"
echo " - cv2.imshow() will NOT work from systemd service"
echo " - Service will auto-start on next boot"
echo "=========================================="
