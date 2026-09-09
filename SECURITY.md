# Security Policy

- Never commit secrets. Use `.env` (git-ignored) and GitHub Secrets.
- All tool execution goes through the runtime validator – the model never receives direct execution privileges.
- Report vulnerabilities privately via GitHub Security Advisories.
- Least-privilege tool design is mandatory for any new tool.
