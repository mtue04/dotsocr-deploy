#!/bin/bash

# One-liner installer for DOTSOCR on fresh Linux GPU server
# Usage: curl -fsSL https://raw.githubusercontent.com/mtue04/dotsocr-deploy/main/install.sh | bash

echo "========================================"
echo "DOTSOCR One-Click Installer"
echo "========================================"
echo

# Clone and setup
git clone https://github.com/mtue04/dotsocr-deploy.git
cd dotsocr-deploy

# Make scripts executable
chmod +x auto_setup.sh run.sh

# Run auto setup
./auto_setup.sh