# Security policy

Do not commit access tokens, passwords, dataset credentials, private download
links, or machine-specific `.env` files. Configure credentials through local
environment variables or the provider's credential manager.

If a credential is committed, revoke it immediately, remove it from Git
history, and rotate it before publishing a new release. Report security issues
privately to the corresponding author rather than opening a public issue.
