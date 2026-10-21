# Resolve open dependency security alerts

1. Snapshot GitHub open Dependabot alerts and verify the locked versions fall in the affected ranges. Check code and secret scanning availability.
2. Upgrade only affected dependencies and their required compatibility partners. Keep Torch and torchvision compatible, align the uv development pin, required version, CI and README bootstrap, and regenerate uv.lock.
3. Check every captured alert against the updated lock, resolve all optional extras, and run Ruff, formatting, Pyright, the CPU test suite and CLI smoke check. Record any validation limits.
4. Review the diff, push a focused PR using the repository template, and inspect remote CI. Alerts close after the fixes reach the default branch and GitHub rescans.
