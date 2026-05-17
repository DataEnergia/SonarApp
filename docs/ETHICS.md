# ETHICS — Senti Ethics Checklist

> This document defines the ethical commitments and known limitations of Senti.
> Required reading for all contributors, agents, and reviewers.

---

## 1. Core ethical commitments

### 1.1 Senti is not stalkerware

Senti is **explicit**, **visible**, and **user-controlled**:

- The app has its own icon and name. It is **not disguised** as a calculator, notepad, or any other app.
- The app shows up in the standard Android app drawer.
- A persistent foreground notification is shown whenever the listening service is running.
- The user can pause or stop the service at any time with one tap.
- There is no "hidden mode" or "stealth installation".

Compare to anti-patterns Senti explicitly avoids:
- App disguised as another type of app (e.g., calculator)
- Continuous background audio capture without user activation
- Hidden processes
- Reverse-trigger UX (e.g., "say a secret phrase to activate")

These patterns are characteristic of stalkerware and are abused by domestic abusers against the very populations they claim to protect. Senti does not adopt any of them.

### 1.2 Audio sovereignty

- Audio is captured **only via the device microphone** when the user has placed the call on **speakerphone**.
- Audio is **never** captured from Android's `VOICE_CALL` audio source. We never request `CALL_PHONE`, `READ_CALL_LOG`, `READ_PHONE_STATE`, or `PROCESS_OUTGOING_CALLS` permissions.
- Audio is held in a **30-second rolling buffer** in volatile memory, continuously overwritten and discarded.
- Audio is **never** persisted to disk, never written to a file, never uploaded.
- Audio is **never** transmitted off the device, neither for inference nor for "model improvement" nor for telemetry.

### 1.3 Decision sovereignty

- Senti **suggests, never decides**. The app never hangs up the call. The app never blocks a contact without explicit user action.
- Every alert uses **probabilistic language**: "possível golpe", "indícios sugerem", "recomendamos". Never "isto é golpe" or "this is fraud".
- Every alert shows **the signals that triggered it**, in plain language. The user sees: "Detectamos: alegação de ser do banco + pressão temporal + pedido de dados pessoais." This is the opposite of a black box.
- Every alert includes a **suggested action that is reversible**: "Desligue e ligue de volta no número oficial." Never "Não pague essa pessoa" (which the app cannot enforce and which assumes guilt).

### 1.4 Auditability

- Every classification is logged locally in Room (SQLite) with: model version, timestamp, transcript, signals detected, confidence, reasoning.
- The user can export the audit log as JSON at any time.
- Defensorias Públicas, Ministério Público, and consumer-protection organizations can audit the full decision history of any user who consents.
- The model and its prompts are open-source under Apache 2.0 / CC-BY 4.0.

### 1.5 Probabilistic communication

The classifier emits a `confidence` value. The UI communicates uncertainty explicitly:

- High confidence + danger → red alert with "POSSÍVEL GOLPE — DESLIGUE"
- Medium confidence + danger → yellow alert with "ATENÇÃO — verifique antes de prosseguir"
- Low confidence → no alert, but signals still shown in detailed view

The product never says "this IS a scam." It says "indicators suggest this MAY be a scam."

---

## 2. Out-of-scope user groups (declared)

Senti is **not designed for** and **MUST NOT be deployed to**:

### 2.1 Victims of domestic violence

The presence of a continuously-listening audio system in a household where one party is an abuser creates serious risks:
- Abuser can use the app to monitor the victim
- Abuser may retaliate violently upon detecting the app
- Speaker phone audio is reachable by any present person, including the abuser
- The product is **not designed to address ambient threats**

Organizations supporting DV survivors should use specialized tools (signal-of-distress apps with covert UX) that have been co-designed with survivor input — not this product.

### 2.2 Minors

The product targets adults age 60+ who consent to its use. Privacy-preserving design for minors is a separate research problem (parental controls, consent capacity, schooling environments).

### 2.3 Professionals under confidentiality obligations

Lawyers, therapists, doctors, journalists, and other professionals with legal confidentiality duties cannot use Senti during professional calls. The product is for personal calls in the user's home.

### 2.4 Jurisdictions with restrictive recording laws

Some jurisdictions (e.g., some US states, some European countries) require two-party consent for any audio recording, even of one's own conversations. Senti's design (speakerphone + microphone, not call recording) probably qualifies as "ambient" rather than "recording", but users in restrictive jurisdictions should consult local law.

---

## 3. Known vulnerabilities, declared honestly

### 3.1 The caregiver-abuser case

**Description.** Caregivers (children, grandchildren, neighbors, hired help) are statistically the **most common source** of financial abuse against elderly people. A caregiver who is the abuser could install Senti, configure themselves as the trusted contact, and use the app to monitor the elder's communications with other family members — or to detect when the elder is being warned by a third party.

**Cannot be fully eliminated.** Any system with a "trusted contact" feature has this attack surface.

**Mitigations:**
- The user (elder) controls the trusted contact list directly. The list is editable from a settings screen that is accessible without entering a password.
- Senti never sends call transcripts to the trusted contact — only metadata alerts ("call classified as suspicious").
- The audit log is local and accessible to the user, and exportable for review by third parties (Defensoria do Idoso, MP).
- The settings screen prominently displays the Disque 100 (Brazil) / equivalent elder-protection hotline number, always one tap away.
- The README and in-app onboarding declare this risk explicitly.

### 3.2 False positives on legitimate calls

**Description.** Real banks call to confirm transactions. Real family members ask for money. Telemarketers use urgency. All of these may share lexical signals with scams.

**Mitigations:**
- Classifier is calibrated (Track D) to bias toward `suspicious` rather than `danger` when not all triggers are present.
- UI always shows the specific signals detected, allowing the user to evaluate the context.
- TTS audio alert (red state) only fires when at least one **critical** signal is detected, not just on accumulated medium-severity signals.
- Suggested action is always "verify via official channel", never "do not engage".

### 3.3 False negatives

**Description.** Adversaries adapt. A scammer who knows Senti exists can adjust wording to evade detection.

**Mitigations:**
- The classifier operates on semantic patterns (intent), not keywords.
- Prompt explicitly trained against rephrasing strategies.
- Open source allows community contribution of new patterns.
- The product is positioned as an **additional layer of defense**, not a replacement for awareness training and skepticism.

### 3.4 Hardware exclusion

**Description.** Senti requires 4GB+ RAM. Many elderly Brazilians, especially in low-income areas, use phones with 2-3GB RAM (Moto E line, Samsung A0x line) where the product cannot run.

**This is a real inequity.** The most vulnerable users are partially excluded.

**Mitigations:**
- Acknowledge openly in the README and write-up.
- Roadmap includes a "companion device" variant: a low-cost Raspberry Pi 5 plugged into the home WiFi network that acts as the inference server for any phone in the house.
- Smaller distilled models are part of the future work plan.

### 3.5 Dependency creation

**Description.** Elderly users may become dependent on Senti and lose the ability to detect scams without it. If the app fails (battery, bug, model error), they may be more vulnerable than before.

**Mitigations:**
- Educational mode after each classified call: "Estes foram os sinais detectados. Por que cada um é suspeito?" — explicit knowledge transfer.
- The app is positioned as a "second opinion", not a primary defense.
- Onboarding includes training material on common scam patterns.

### 3.6 Cultural and linguistic blindspots

**Description.** Scams against indigenous, immigrant, and dialect-using elders may use cultural patterns the model does not capture. The training data and validation dataset are biased toward standard PT-BR and EN-US.

**Mitigations:**
- Acknowledged in limitations section of the write-up.
- Open source allows community contribution of regional patterns.
- Future work: low-resource language adapters.

---

## 4. Compliance check

### 4.1 LGPD (Brazil)

| Requirement | Status |
|---|---|
| Art. 7, VII — protection of life | Aligned (the app exists to protect vulnerable users) |
| Art. 11 — sensitive personal data | Audio is sensitive; never leaves device; opt-in only |
| Art. 18 — user rights | User can access all data, export, delete |
| Art. 46 — security measures | No transmission, no persistence of audio, local-only |
| Privacy policy required | Yes, distributed with the app |

### 4.2 Google Play Policy

| Policy | Status |
|---|---|
| Restricted permissions (Call Log, SMS) | Not requested |
| Spyware policy | Not applicable — app is overt, not covert |
| Sensitive permissions disclosure | `RECORD_AUDIO` clearly disclosed at first use |
| User Data policy | Compliant — no data exfiltration |

### 4.3 GDPR (Europe — future)

| Requirement | Status |
|---|---|
| Lawful basis (Art. 6) | Consent (explicit, in-app) |
| Special categories (Art. 9) | Audio of phone calls — explicit consent, local processing |
| Right to erasure (Art. 17) | Implementable — local data, one-tap delete |
| Data Protection by Design (Art. 25) | Core architecture is privacy-preserving |
| Cross-border transfer | None — data never leaves device |

### 4.4 Apache 2.0 + CC-BY 4.0

| Component | License | Compatible |
|---|---|---|
| Senti code | Apache 2.0 | Yes |
| Senti docs / dataset | CC-BY 4.0 | Yes |
| Gemma 4 weights | Gemma Terms of Use | Yes (commercial use allowed) |
| MediaPipe | Apache 2.0 | Yes |
| whisper.cpp | MIT | Yes |
| Coqui TTS / Piper (for dataset) | MPL-2.0 / MIT | Yes |

---

## 5. Acknowledgments and review invitations

We acknowledge that this product addresses a sensitive domain (elder protection) where ethics is not optional and where well-intentioned but poorly-designed systems can cause harm.

We explicitly invite review and contribution from:

- **Brazilian elder-protection institutions**: Ministério Público (Promotoria do Idoso), Defensoria Pública (Núcleo do Idoso), CRAS, CREAS, Conselhos do Idoso
- **NGOs**: SBGG (Sociedade Brasileira de Geriatria e Gerontologia), HelpAge International
- **Academic researchers**: Cornell IPV-Spyware research group, NIST elder-fraud working groups
- **Regulators**: SENACON, ANATEL, INSS Ouvidoria
- **Disability rights advocates**: especially around UI accessibility for low vision and motor impairments

We commit to revisiting these ethics commitments based on input from those communities.
