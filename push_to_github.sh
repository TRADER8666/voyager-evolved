#!/bin/bash
#
# push_to_github.sh - Push Voyager Evolved to GitHub
#
# Usage: ./push_to_github.sh YOUR_GITHUB_TOKEN
#
# To create a Personal Access Token:
# 1. Go to https://github.com/settings/tokens/new
# 2. Give it a name like "voyager-evolved-push"
# 3. Select expiration (e.g., 30 days)
# 4. Check these scopes: repo (full control of private repos)
# 5. Click "Generate token" and copy it
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

REPO_NAME="voyager-evolved"
GITHUB_USER="TRADER8666"

# Check if token is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: GitHub token required${NC}"
    echo ""
    echo "Usage: $0 YOUR_GITHUB_TOKEN"
    echo ""
    echo "To create a Personal Access Token with repo creation permissions:"
    echo "  1. Go to: https://github.com/settings/tokens/new"
    echo "  2. Name: voyager-evolved-push"
    echo "  3. Expiration: 30 days (or your preference)"
    echo "  4. Scopes: Check 'repo' (Full control of private repositories)"
    echo "  5. Click 'Generate token' and copy it"
    echo ""
    echo "Then run: $0 ghp_YOUR_TOKEN_HERE"
    exit 1
fi

TOKEN="$1"

echo -e "${YELLOW}=== Voyager Evolved GitHub Push Script ===${NC}"
echo ""

# Verify token works
echo -e "${YELLOW}Verifying GitHub token...${NC}"
USER_CHECK=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.github.com/user)
if echo "$USER_CHECK" | grep -q '"login"'; then
    ACTUAL_USER=$(echo "$USER_CHECK" | grep '"login"' | head -1 | cut -d'"' -f4)
    echo -e "${GREEN}✓ Token valid for user: $ACTUAL_USER${NC}"
else
    echo -e "${RED}✗ Invalid token or authentication failed${NC}"
    echo "$USER_CHECK"
    exit 1
fi

# Check if repo exists
echo -e "${YELLOW}Checking if repository exists...${NC}"
REPO_CHECK=$(curl -s -H "Authorization: Bearer $TOKEN" "https://api.github.com/repos/$ACTUAL_USER/$REPO_NAME")
if echo "$REPO_CHECK" | grep -q '"full_name"'; then
    echo -e "${GREEN}✓ Repository already exists${NC}"
else
    # Create repository
    echo -e "${YELLOW}Creating repository: $REPO_NAME...${NC}"
    CREATE_RESULT=$(curl -s -X POST \
      -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" \
      https://api.github.com/user/repos \
      -d "{\"name\":\"$REPO_NAME\",\"description\":\"Voyager Evolved - An enhanced Minecraft AI agent with personality and social awareness\",\"private\":false,\"auto_init\":false}")
    
    if echo "$CREATE_RESULT" | grep -q '"full_name"'; then
        echo -e "${GREEN}✓ Repository created successfully${NC}"
    else
        echo -e "${RED}✗ Failed to create repository${NC}"
        echo "$CREATE_RESULT"
        exit 1
    fi
fi

# Configure git
echo -e "${YELLOW}Configuring git...${NC}"
git config user.email "trader8666@users.noreply.github.com" 2>/dev/null || true
git config user.name "TRADER8666" 2>/dev/null || true

# Set up remote
echo -e "${YELLOW}Setting up remote...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin "https://${TOKEN}@github.com/${ACTUAL_USER}/${REPO_NAME}.git"

# Push code
echo -e "${YELLOW}Pushing code to GitHub...${NC}"
git push -u origin main --force

# Clean up token from remote (security)
git remote set-url origin "https://github.com/${ACTUAL_USER}/${REPO_NAME}.git"

echo ""
echo -e "${GREEN}=== Success! ===${NC}"
echo -e "Repository URL: ${GREEN}https://github.com/${ACTUAL_USER}/${REPO_NAME}${NC}"
echo ""
echo "Next steps:"
echo "  1. Visit your repository: https://github.com/${ACTUAL_USER}/${REPO_NAME}"
echo "  2. Add topics: minecraft, ai-agent, voyager, openai"
echo "  3. Enable GitHub Actions in repository settings"
echo ""
