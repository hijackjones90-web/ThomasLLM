#!/usr/bin/env bash
# Runpod setup for Kilo Code — follow https://agents.runpod.io/
# Run this script from your project root. Requires Node.js 22.20+.
set -euo pipefail

echo "=== Step 1: Install Runpod skills for Kilo Code ==="
npx -y skills add runpod/runpod-plugins-official \
  --skill '*' \
  --agent kilo \
  --yes \
  --global

echo ""
echo "=== Step 2: Verify skills installed ==="
npx -y skills list --global

echo ""
echo "=== Step 3: Register Runpod MCP server (Kilo Code) ==="
# Kilo Code reads MCP config from .kilo/mcp.json (project) or ~/.kilo/mcp.json (user).
# If you prefer CLI registration, run:
#   npx @runpod/mcp-server@latest add --client kilo
# If that fails because 'kilo' isn't a recognized --client id, the config file
# in .kilo/mcp.json (already written) is sufficient.

echo ""
echo "=== Step 4: Reload Kilo Code ==="
echo "Restart Kilo Code (or run /reload-plugins) so it picks up the MCP server."
echo "Then sign in with OAuth the first time you call a Runpod tool."

echo ""
echo "=== Step 5: Verify end-to-end ==="
echo "List your Pods to confirm the MCP connection works:"
echo "  npx -y @runpod/mcp-server@latest pod list"

echo ""
echo "Setup complete. If OAuth fails, see the guide's 'If OAuth sign-in fails' section."