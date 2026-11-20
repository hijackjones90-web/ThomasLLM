# Runpod Setup for Kilo Code

Followed the official guide at https://agents.runpod.io/ (also at https://docs.runpod.io/agent-setup.md).

## What was done

1. **MCP server config written** to `.kilo/mcp.json`:
   ```json
   {
     "mcpServers": {
       "runpod": { "url": "https://mcp.runpod.io/" }
     }
   }
   ```

2. **Setup script created** at `setup_runpod.sh` with the exact commands from the guide.

## Commands you must run manually

I cannot execute `npx` or modify your global agent config from this sandbox. Run these:

```bash
# 1. Install the Runpod skills (agent id: kilo)
npx -y skills add runpod/runpod-plugins-official \
  --skill '*' \
  --agent kilo \
  --yes \
  --global

# 2. Verify
npx -y skills list --global

# 3. Register MCP server (optional — .kilo/mcp.json already has it)
npx @runpod/mcp-server@latest add --client kilo

# 4. Reload Kilo Code so it picks up the new skills + MCP server
#    (restart the IDE or run /reload-plugins)

# 5. Sign in with OAuth the first time you call a Runpod tool
#    (browser window opens automatically)

# 6. Verify end-to-end by listing Pods
npx -y @runpod/mcp-server@latest pod list
```

## Expected result

```
┌─ Runpod Agent Setup ─────────────────────────────────┐
│  ✓ Runpod skills   router + lane + journey skills    │
│  ⚠ runpod MCP      needs sign-in: /mcp → runpod      │
│                                                      │
│  ⚡ Reload or restart your agent to load the skills  │
└──────────────────────────────────────────────────────┘
```

## If OAuth fails

Use an API key instead:

1. Create one at https://console.runpod.io/user/settings
2. Export it: `export RUNPOD_API_KEY=rk_...`
3. Update `.kilo/mcp.json`:
   ```json
   {
     "mcpServers": {
       "runpod": {
         "url": "https://mcp.runpod.io/",
         "headers": { "Authorization": "Bearer ${RUNPOD_API_KEY}" }
       }
     }
   }
   ```

## Docs feedback

The guide says to report back through the Runpod docs MCP server (`submit_feedback`).
If that tool is available after setup, send one entry per page read:
- `/agent-setup` — POSITIVE: clear install commands and agent-id table.
- `/serverless/overview` — not read yet; will report after first deploy.

## Next

Once Runpod is connected, I can use the Runpod MCP tools to:
- Discover GPU templates (A100, H100, RTX 4090, etc.)
- Deploy a Pod with ≥24 GB VRAM for ThomasVLM training
- Run `bash run_all.sh` inside the Pod
- Pull trained artifacts back to `/artifacts/thomasvlm/`

Say "deploy the training pod" when you're ready.