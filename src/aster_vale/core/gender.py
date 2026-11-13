"""Rule-based caller / patient gender inference (no transformers, no ML model).

Weighted evidence voting over transcript cues:
  1. How the AGENT addresses the customer (sir / ma'am / bhaiya / didi ...)   - strongest, frequent
  2. Customer self-kinship ("my husband" -> female caller, "my wife" -> male caller)
  3. The customer's stated first name, looked up in a small Indian first-name lexicon (+ suffix backoff)
  4. Who the patient is ("my father" -> male patient, "my mother" -> female patient, "for myself" -> caller)
Each cue adds signed log-odds evidence (+ male, - female). |score| below a threshold -> "unknown".
"""

from __future__ import annotations

import math
import re

# --- lexicons ------------------------------------------------------------------
MALE_ADDRESS = {"sir": 1.0, "sirji": 1.0, "bhaiya": 1.2, "bhai": 0.8, "uncle": 1.0, "mr": 1.2, "boss": 0.6,
                "bhaisahab": 1.2}
FEMALE_ADDRESS = {"ma'am": 1.0, "maam": 1.0, "mam": 1.0, "madam": 1.0, "madame": 1.0, "didi": 1.2,
                  "aunty": 1.0, "auntie": 1.0, "behenji": 1.2, "mrs": 1.2, "miss": 1.0, "ms": 1.0}

MALE_KIN = r"father|dad|daddy|papa|husband|brother|son|grandfather|grandpa|nana|dada|uncle|father-in-law|chacha|mama"
FEMALE_KIN = (r"mother|mom|mummy|mum|mumma|maa|wife|sister|daughter|grandmother|grandma|nani|dadi|aunt|aunty|"
              r"mother-in-law|bhabhi|chachi|mami")

# Frequent Indian first names (illustrative open lexicon; extend via data/names_*.txt if desired).
MALE_NAMES = set("""
aarav abhay abhishek aditya ajay akash akshay alok aman amit amitabh anand anil ankit anuj arjun arun ashok
ashish atul ayaan bharat chetan deepak dev devansh devendra dhruv dinesh farhan gaurav gopal hamid harish
hemant imran irfan jatin kabir kamal karan kartik kunal lalit mahesh manish manoj mehul mohan mohit mukesh
nalin naman naren naresh naveen navin neel nikhil nitin pankaj paras pradeep pramod pranav prakash prateek
rahul raj rajesh rajiv rakesh ramesh ravi rayan rishi ritesh rohan rohit sachin sahil salman sameer sandeep
sanjay santosh saurabh shivam shyam siddharth sudhir sumit sunil suresh tarun tushar umesh varun vedant veer
vijay vikas vikram vinay vinod vishal vivek yash yogesh zayan zubair harsh pritam puneet raghav kishore
""".split())
FEMALE_NAMES = set("""
aarti aisha alina amrita anaya anita anjali ankita anushka archana asha ayesha bhavna deepa deepika divya
farina farzana fatima gauri geeta isha jaya jyoti kajal kavita kiran komal kusuma lakshmi leela madhu mamta
meena meera mira nalini neha nidhi nikita nisha nitu pallavi parveen pilasha pooja poonam preeti priya
priyanka radha rani reena rekha reva rina ritu roshni ruchi sana sangeeta sapna sarita saira seema shalini
shanti shilpa shweta simran sneha sonal sonia sunita sushma swati tanvi tara usha vandana varsha vidya
zainab zoya fariha sakshi tanya
""".split())
MALE_ENDINGS = ("esh", "endra", "deep", "jeet", "jit", "veer", "pal", "raj", "an", "ant", "ush", "it", "ul")
FEMALE_ENDINGS = ("a", "i", "ee", "ika", "ini", "ita", "shree", "devi", "bai", "ben")

SPEAKER_RE = re.compile(r"^\s*(agent|customer)\s*:\s*(.*)$", re.I)
NAME_INTRO_RE = re.compile(r"\b(?:my name is|i am|this is|i'm)\s+([A-Z][a-z]+)", re.I)
ADDRESSED_NAME_RE = re.compile(r"(?:,\s*|\b(?:okay|ok|yes|so|hello)\s+)([A-Z][a-z]{2,})(?:\s+(?:ji|sir|ma'am)\b|\?|,)")
NAME_Q_RE = re.compile(r"\b(?:your (?:good )?name|what is your name|may i know your name)\b", re.I)
SELF_PATIENT_RE = re.compile(r"\b(?:for myself|for me|i have (?:pain|a problem)|my (?:knee|back|leg|neck|"
                             r"shoulder|spine|hip|foot|heel|hand) )", re.I)
# "I am Renu's husband speaking" -> caller's own gender from the stated relation; patient is the named person.
CALLER_ROLE_RE = re.compile(r"\b(?:i am|i'm|this is)\s+(?:[A-Z][a-z]+(?:'s)?\s+)?(?:the\s+)?(husband|wife|son|daughter|"
                            r"father|mother|brother|sister)\b", re.I)
ROLE_SIGN = {"husband": 4.0, "son": 4.0, "father": 4.0, "brother": 4.0,
             "wife": -4.0, "daughter": -4.0, "mother": -4.0, "sister": -4.0}
THRESHOLD = 0.5  # one explicit agent address term (min recency weight 0.5) is enough evidence


def _turns(transcript: str) -> list[tuple[str, str]]:
    out = []
    for line in transcript.splitlines():
        m = SPEAKER_RE.match(line)
        if m:
            out.append((m.group(1).lower(), m.group(2)))
    return out


def name_gender(name: str) -> float:
    n = name.lower().strip(".,")
    if n in MALE_NAMES:
        return 2.0
    if n in FEMALE_NAMES:
        return -2.0
    if n.endswith(MALE_ENDINGS):
        return 0.6
    if n.endswith(FEMALE_ENDINGS):
        return -0.6
    return 0.0


def _address_score(text: str) -> float:
    s = 0.0
    words = re.findall(r"[a-z']+", text.lower())
    for w in words:
        s += MALE_ADDRESS.get(w, 0.0)
        s -= FEMALE_ADDRESS.get(w, 0.0)
    return s


def _label(score: float) -> str:
    if score >= THRESHOLD:
        return "male"
    if score <= -THRESHOLD:
        return "female"
    return "unknown"


def _confidence(score: float) -> float:
    return round(1 / (1 + math.exp(-abs(score) / 2)), 3)


def predict_gender(transcript: str) -> dict:
    turns = _turns(transcript)
    n = max(len(turns), 1)
    caller, patient = 0.0, 0.0
    male_kin, female_kin = 0, 0
    cues: list[str] = []
    patient_is_caller = False
    patient_override: str | None = None
    names: list[str] = []

    for i, (spk, text) in enumerate(turns):
        recency = 0.5 + i / n  # later turns weigh more (final, settled form of address)
        low = text.lower()
        if spk == "agent":
            s = _address_score(text) * recency
            if s:
                caller += s
            for am in ADDRESSED_NAME_RE.finditer(text):
                if name_gender(am.group(1)) and abs(name_gender(am.group(1))) == 2.0:
                    names.append(am.group(1))
            # Agent asked for the customer's name -> next customer turn is often just the name.
            if NAME_Q_RE.search(text) and i + 1 < len(turns) and turns[i + 1][0] == "customer":
                reply = re.findall(r"[A-Z][a-z]+", turns[i + 1][1])
                if reply and not re.search(r"\b(father|mother|patient|his|her)\b", text, re.I):
                    names.append(reply[0])
        else:  # customer
            rm = CALLER_ROLE_RE.search(text)
            if rm:
                caller += ROLE_SIGN[rm.group(1).lower()]
                cues.append(f"caller introduces self as {rm.group(1).lower()}")
                spouse = {"husband": "female", "wife": "male"}.get(rm.group(1).lower())
                if spouse:
                    patient_override = spouse  # "I am Renu's husband" -> the patient is his wife
            if re.search(r"\bmy husband\b", low):
                caller -= 3.0
                cues.append("caller says 'my husband'")
            if re.search(r"\bmy wife\b", low):
                caller += 3.0
                cues.append("caller says 'my wife'")
            m = NAME_INTRO_RE.search(text)
            if m and m.group(1).lower() not in ("calling", "speaking", "from", "not", "the", "a"):
                names.append(m.group(1))
            for km in re.finditer(rf"\bmy ({MALE_KIN})\b", low):
                if km.group(1) != "husband":
                    patient += 1.5
                    male_kin += 1
                    cues.append(f"patient: 'my {km.group(1)}'")
            for km in re.finditer(rf"\bmy ({FEMALE_KIN})\b", low):
                if km.group(1) != "wife":
                    patient -= 1.5
                    female_kin += 1
                    cues.append(f"patient: 'my {km.group(1)}'")
            if SELF_PATIENT_RE.search(text):
                patient_is_caller = True

    # Prefer a lexicon-exact name, then a suffix-only guess; first mention wins within a tier.
    ranked = sorted(names, key=lambda x: -abs(name_gender(x)))
    stated_name = ranked[0] if ranked else None
    if stated_name:
        ns = name_gender(stated_name)
        caller += ns
        if ns:
            cues.append(f"stated name '{stated_name}' ({'male' if ns > 0 else 'female'} lexicon)")
    if caller:
        cues.append(f"agent address terms score {caller:+.1f}")

    # Patient: explicit relative wins; otherwise assume the caller is the patient.
    if patient_override:
        patient_label, patient_src = patient_override, "spouse"
    elif male_kin >= 2 and female_kin >= 2:
        patient_label, patient_src = "multiple", "relatives"
    elif abs(patient) < THRESHOLD or (patient_is_caller and abs(patient) < 3):
        patient_label, patient_src = _label(caller), "caller"
    else:
        patient_label, patient_src = _label(patient), "relative"
    return {
        "caller_gender": _label(caller),
        "caller_confidence": _confidence(caller),
        "patient_gender": patient_label,
        "patient_is": patient_src,
        "caller_score": round(caller, 2),
        "cues": cues[:8],
    }
