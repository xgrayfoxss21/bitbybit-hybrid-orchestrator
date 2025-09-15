# SECURITY.md

**Project:** BitByBit Hybrid Orchestrator  
**Owner:** **Foxes Den Corp (Grayfox)**  
**Version:** **alpha v0.002**  
**License:** AGPL-3.0-or-later

This document explains how to report security issues and how we coordinate fixes and disclosure.

---

## 📬 Contact Summary

- **Report privately (preferred):** GitHub → **Security** → **Report a vulnerability**  
  <https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/security/advisories/new>
- **Email (fallback):** troubleshooting@foxesden.xyz
- **PGP public key:** [`security/pgp/xgrayfoxss21.asc`](security/pgp/xgrayfoxss21.asc)

### 🔐 PGP

- **Maintainer:** Foxes Den Corp (Grayfox)  
- **Email:** troubleshooting@foxesden.xyz  
- **Key ID (long):** *(to be published after key upload)*  
- **Fingerprint:** *(to be published after key upload)*  
- **Algorithms:** Ed25519 (sign) + Curve25519 (encrypt)  
- **Public key (.asc):** [`security/pgp/xgrayfoxss21.asc`](security/pgp/xgrayfoxss21.asc)

**Verify & import**
```bash
gpg --show-keys --with-fingerprint security/pgp/xgrayfoxss21.asc
# Confirm the fingerprint matches exactly (spaces optional).

gpg --import security/pgp/xgrayfoxss21.asc
````

**Encrypt a report (optional but recommended)**

```bash
# Prepare your report (include details below).
gpg --encrypt --armor -r troubleshooting@foxesden.xyz report.txt
# Email the resulting ASCII file.
```

---

## 🧭 What to include in a report

Please provide as much of the following as possible:

* A clear **overview** and **impact** (why this matters).
* **Affected version/commit** (link the exact commit or tag).
* **Steps to reproduce** (minimal PoC preferred).
* Expected vs. actual behavior.
* Any **logs/stack traces** (scrub secrets/PII).
* Suggested **mitigations** (if known).
* Your **disclosure preference** (credit or anonymous).

> ⚠️ Do **not** include real personal data or credentials. Use synthetic examples—the project ships with PII redaction features, but treat reports as if they will be shared among maintainers.

---

## 🎯 Scope

**In scope**

* Code and configurations in this repository.
* The published documentation site (e.g., [https://bit.foxesden.xyz/](https://bit.foxesden.xyz/) or GitHub Pages for this repo).
* Demo UIs (Colab/Gradio) and the orchestrator flow as implemented here.

**Out of scope**

* Vulnerabilities exclusively in **third-party dependencies** without a demonstrable exploit path via this project.
* **Social engineering**, **physical attacks**, or **DDoS** / volumetric resource exhaustion.
* Automated scans causing service degradation on shared demo endpoints.
* License/compliance questions (see **COMPLIANCE.md**); these are important but not security bugs.

---

## 🤝 Coordinated disclosure policy

We follow responsible, time-boxed disclosure with collaborative timelines.

**Acknowledgment**
We aim to acknowledge your report **within 48 hours**.

**Triage & assessment**
Severity determined using CVSS 3.x/4.0 as guidance.

**Target timelines** (adjusted as needed by complexity):

* **Critical/High:** fix or viable mitigation within **14–30 days**.
* **Medium:** **30–60 days**.
* **Low/Informational:** **60–90 days**.

**Public disclosure**
We coordinate a disclosure date once a patch/mitigation is available. If a fix is delayed, we’ll provide status updates and may agree on a partial disclosure.

**Credit**
We will credit researchers in **CHANGELOG.md** and/or the advisory unless you request anonymity.

> We currently do **not** run a paid bug bounty. Thoughtful reports are still highly appreciated, and we’re happy to acknowledge your contribution.

---

## 🛡️ Safe-harbor statement

If you make a **good-faith** effort to follow this policy:

* We will **not pursue legal action** against you for security research.
* Exclusions: data exfiltration, privacy violations, permanent service impact, extortion, or breach of applicable laws.
* Stop testing and report immediately if you encounter **user data**, credentials, or PII.

---

## 🔒 Handling sensitive data

* Prefer **sanitized** inputs and outputs in your PoC.
* If you must share sensitive details, **encrypt** via the maintainer PGP key.
* We avoid storing raw prompts/outputs; demo UIs are ephemeral. If logs are necessary, store **redacted** content only.

---

## 🧪 Patch, release, and advisory

* Fixes land on `main` with tests where applicable.
* A security **advisory** summarizes impact, affected versions/commits, and upgrade/migration steps.
* The **CHANGELOG** will reference the advisory and acknowledge reporters (if desired).
* For qualifying issues, we may request a CVE via GitHub Security Advisories.

---

## 🧾 Compliance notes (AGPL §13)

If you deploy this project over a network (even privately), the AGPL requires exposing the **Corresponding Source** for the running commit. We provide **COMPLIANCE.md** with:

* HTTP header:
  `X-AGPL-Source: https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/tree/<COMMIT_SHA>`
* `/source` endpoint & UI footer snippets

Security issues stemming solely from missing AGPL headers/links are **compliance** matters; please open a (private) advisory or issue accordingly.

See: **[COMPLIANCE.md](COMPLIANCE.md)**

---

## 🔁 Key rotation & verification

If the maintainer key rotates or expires:

* A new `.asc` will be added under `security/pgp/` and the old key will be kept until its expiration (with a revocation certificate if applicable).
* **SECURITY.md** will be updated with the new fingerprint.
* Verify keys with:

  ```bash
  gpg --show-keys --with-fingerprint security/pgp/xgrayfoxss21.asc
  ```

---

## 🧰 Deployment hardening tips (alpha v0.002)

While this is a security policy (not a hardening guide), the following practical defaults help reduce risk during **alpha**:

* **Disable public Gradio sharing in production** (`share=False`) and place behind a reverse proxy with TLS.
* **Enable auth** (proxy auth, mTLS, or app-layer tokens) for API/UI; rotate secrets regularly.
* **Least privilege:** run under a non-root user; restrict filesystem permissions to data dirs.
* **Rate limiting & size caps:** configure per-IP/min quotas and request size limits.
* **Outbound controls:** restrict egress if your deployment doesn’t require internet.
* **Log redaction:** ensure PII/redaction is enabled in logs; avoid storing raw prompts/outputs.
* **Supply-chain checks:** pin dependencies, run `pip-audit`/`safety`, review new transitive deps.

---

## 📎 Templates & references

* Private advisories: [https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/security/advisories/new](https://github.com/xgrayfoxss21/bitbybit-hybrid-orchestrator/security/advisories/new)
* PGP key: [`security/pgp/xgrayfoxss21.asc`](security/pgp/xgrayfoxss21.asc)
* CHANGELOG: [`CHANGELOG.md`](CHANGELOG.md)
* Compliance: [`COMPLIANCE.md`](COMPLIANCE.md)

Thank you for helping keep **BitByBit Hybrid Orchestrator** safe for everyone!

```

If you want, I can also drop this into the canvas doc on your next message so it replaces the current version exactly.
::contentReference[oaicite:0]{index=0}
```
