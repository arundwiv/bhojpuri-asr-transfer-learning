#!/usr/bin/env python3
"""
Generate QLoRA fine-tuning corpus for Bhojpuri AI assistant.

Output: eval-monitoring/finetune_corpus_v1.jsonl
Format: TRL SFTTrainer messages format —
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage (Colab):
  !pip install -q anthropic
  import os; os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
  # or store in Colab Secrets as ANTHROPIC_API_KEY
  !python eval-monitoring/generate_finetune_corpus.py

Usage (local):
  pip install anthropic
  ANTHROPIC_API_KEY=sk-ant-... python eval-monitoring/generate_finetune_corpus.py

Flags:
  --dry-run    Write gold examples only, skip API generation calls.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Production system prompt — keep in sync with build_bhojpuri_messages()
# in backend-api/app.py
# ---------------------------------------------------------------------------
PROD_SYSTEM_PROMPT = (
    "तू एगो भोजपुरी सहायक बाड़s। "
    "हर जवाब सुद्ध भोजपुरी में दे — हिंदी में बिल्कुल ना। "
    "जवाब 1 से 3 छोट वाक्य में दे। "
    "बेकार भूमिका मत लिख। "
    "एकही बात बार-बार मत दोहरा। "
    "अंग्रेजी शब्द कम से कम रख। "
    "जवाब में 'जवाब:' मत लिख। "
    "खेती से जुड़ल सवाल में सीधा, छोट आ काम के सलाह दे। "
    "लाइव जानकारी जइसे मौसम, बाजार भाव, खबर न होखे त साफ कह दे — अंदाजा मत लगा।"
)

# ---------------------------------------------------------------------------
# 20 verified gold examples — native speaker reviewed 2026-05-28
# Each tuple: (category, question, answer)
# ---------------------------------------------------------------------------
GOLD_EXAMPLES = [
    # cv — conversational
    ("cv", "राम राम! का हाल बा?",
     "राम राम! हम बढ़िया बानी। रउरा कइसन बाड़ीं? का मदद चाही?"),
    ("cv", "कइसन बाड़s?",
     "जोहार! हम नीक बानी। बोलीं, हम का मदद करी?"),
    ("cv", "बहुत धन्यवाद, बड़ा मदद भइल।",
     "एहमें का धन्यवाद! रउरे काम आइल त हमार खुशी बा।"),
    ("cv", "ठीक बा, अब जात बानी।",
     "ठीक बा! जब मन करे, पूछत रहीं। राम राम!"),
    ("cv", "तू का-का कर सकत बाड़s?",
     "हम मौसम, खेती, स्वास्थ्य आ रोजमर्रा के सवाल के जवाब दे सकत बानी। पूछीं!"),
    # gq — general knowledge
    ("gq", "भारत के राजधानी कहाँ बा?",
     "भारत के राजधानी नई दिल्ली बा।"),
    ("gq", "बिहार के मुख्यमंत्री कवन बाड़न?",
     "बिहार के मुख्यमंत्री नीतीश कुमार बाड़न।"),
    ("gq", "भारत के प्रधानमंत्री कवन बाड़न?",
     "भारत के प्रधानमंत्री नरेन्द्र मोदी बाड़न।"),
    ("gq", "सूरज कवन दिशा में उगेला?",
     "सूरज पूरुब दिशा में उगेला आ पच्छिम में डूबेला।"),
    ("gq", "पानी केतना डिग्री पर उबलेला?",
     "पानी 100 डिग्री सेल्सियस पर उबलेला।"),
    ("gq", "महात्मा गाँधी के जनम कब भइल रहे?",
     "महात्मा गाँधी के जनम 2 अक्टूबर सन् 1869 में भइल रहे।"),
    ("gq", "गंगा नदी कहाँ से निकलेली?",
     "गंगा नदी उत्तराखण्ड में गंगोत्री ग्लेशियर से निकलेली आ बंगाल की खाड़ी में मिलेली।"),
    # ag — agriculture
    ("ag", "धान के पत्ता पीयर काहें हो जाला?",
     "धान के पत्ता पीयर होला जब खेत में नाइट्रोजन के कमी होला। "
     "यूरिया खाद छिड़कऽ आ पानी के सही इंतजाम करऽ — कुछ दिन में हरियर हो जाई।"),
    ("ag", "गेहूँ के बोआई कब करे के चाही?",
     "गेहूँ के बोआई नवम्बर के बीच से दिसम्बर सुरुआत तक करे के चाही — "
     "एही समय पैदावार सबसे नीक होला।"),
    ("ag", "आलू के खेती में कवन खाद देवे के चाही?",
     "आलू में डीएपी आ पोटाश खाद दिहल जाला। "
     "बोआई से पहिले खेत में गोबर खाद मिलाव, पैदावार बढ़ी।"),
    # he — health
    ("he", "बुखार में का करे के चाही?",
     "खूब आराम करऽ, पानी आ नींबू पानी पीत रहऽ, हल्का खाना खाव। "
     "दू-तीन दिन से ज्यादा बुखार रहे त डॉक्टर से मिलऽ।"),
    # gs — government schemes
    ("gs", "पीएम किसान योजना का हऊ?",
     "पीएम किसान सम्मान निधि में छोट किसान भाइन के सरकार हर साल 6000 रुपिया देला — "
     "तीन किस्त में, 2000-2000 रुपिया। पइसा सीधे बैंक खाता में आवेला।"),
    ("gs", "आयुष्मान भारत योजना का बा?",
     "एह योजना में गरीब परिवार के लोगन के हर साल पाँच लाख रुपिया तक के "
     "इलाज सरकारी आ चुनल अस्पताल में मुफ्त मिलेला।"),
    # lc — local / culture
    ("lc", "भोजपुरी भाषा कहाँ-कहाँ बोलल जाला?",
     "भोजपुरी मुख्य रूप से बिहार के पच्छिमी जिलन में, पूर्वी उत्तर प्रदेश में "
     "आ झारखण्ड के कुछ हिस्सन में बोलल जाला।"),
    # re — refusal / no live data
    ("re", "आजु बाजार में सरसों के भाव का बा?",
     "हमरा लगे आजुके सरसों के लाइव भाव नइखे। नजदीकी मंडी में पता करऽ।"),
]

# ---------------------------------------------------------------------------
# Per-category generation targets (additional examples beyond gold)
# Total: 20 gold + 129 generated = 149 examples
# ---------------------------------------------------------------------------
CATEGORIES = [
    ("cv", "Conversational — greetings, thanks, farewells, assistant capabilities, "
           "follow-up questions", 15),
    ("gq", "General knowledge — geography, science facts, history, Indian leaders, "
           "states and capitals, rivers, animals", 23),
    ("ag", "Agriculture — specific crops (paddy, wheat, sugarcane, vegetables, mustard), "
           "fertilizers, pesticides, irrigation, harvesting, storage, pests", 32),
    ("he", "Health — common ailments (cough, cold, diarrhoea, eye problems, snake bite), "
           "first aid, nutrition, when to see a doctor", 19),
    ("gs", "Government schemes — MGNREGA, Jan Dhan, ration card, PM Awas, Ujjwala, "
           "Fasal Bima, KCC, PM SVANidhi, Ladli Laxmi", 13),
    ("lc", "Local culture — Chhath, Holi, festivals, folk music (Sohar, Birha), "
           "Bhojpuri food, Bhojpur region history", 8),
    ("mu", "Math and units — simple arithmetic, weights (quintal, maund, tola), "
           "land measures (bigha, kattha, dhur, acre), time, currency", 8),
    ("re", "Refusal / no live data — today's mandi prices, live weather, breaking news, "
           "current stock prices; model must decline and redirect clearly", 11),
]

OUTPUT_FILE = Path(__file__).parent / "finetune_corpus_v1.jsonl"
MAX_RETRIES = 3
BATCH_SIZE = 15  # max examples per API call


def make_training_record(q: str, a: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": PROD_SYSTEM_PROMPT},
            {"role": "user",      "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def build_cached_generation_prompt() -> str:
    """
    Static block placed in the system message with cache_control=ephemeral.
    Contains generation rules + all 20 gold examples. Cached across all
    generation calls to save tokens.
    """
    lines = [
        "You generate Bhojpuri fine-tuning Q&A pairs for a rural voice AI assistant.",
        "",
        "BHOJPURI LANGUAGE RULES — apply to every answer:",
        "  Pronoun  : हम (not मैं)",
        "  Copula   : बा / बाड़ीं / बाड़न / बानी (not है / हैं / हूं)",
        "  Negative : ना / नइखे (not नहीं)",
        "  Good     : नीक (not अच्छा)",
        "  Why      : काहें (not क्यों)",
        "  Who/which: कवन (not कौन/क्या)",
        "  How much : केतना (not कितना)",
        "  East/West: पूरुब / पच्छिम (not पूर्व / पश्चिम)",
        "  Habitual : verb + -एला / -एली (e.g., उगेला, मिलेली)",
        "  Imperative: verb stem + ऽ (करऽ, पीऽ) or -व (मिलाव)",
        "  Length   : 1–3 short sentences per answer",
        "  No drift : check every verb ending — Hindi verb endings are errors",
        "",
        "GOLD EXAMPLES (all verified by a native Bhojpuri speaker):",
    ]
    for i, (cat, q, a) in enumerate(GOLD_EXAMPLES, 1):
        lines.append(f"[{i:02d}·{cat}]")
        lines.append(f"Q: {q}")
        lines.append(f"A: {a}")
        lines.append("")
    lines.append(
        "Generate new examples that match this quality and language register exactly."
    )
    return "\n".join(lines)


def parse_json_pairs(text: str) -> list[tuple[str, str]]:
    """Extract a JSON array of {q, a} objects from model output."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    data = json.loads(text)
    return [(item["q"].strip(), item["a"].strip()) for item in data]


def generate_batch(client, cached_block: str, cat: str, desc: str, n: int) -> list[tuple[str, str]]:
    user_prompt = (
        f"Generate exactly {n} NEW Bhojpuri Q&A pairs for category: "
        f"{cat.upper()} — {desc}.\n\n"
        "Requirements:\n"
        "  - Questions must sound like a real rural user speaking into a voice assistant\n"
        "  - No question may duplicate any gold example above\n"
        "  - Every answer must be 100% Bhojpuri — zero Hindi verb endings\n"
        "  - Agriculture: give practical, specific advice (variety names, dosage, timing)\n"
        "  - Government schemes: include real benefit amounts and eligibility hints\n"
        "  - Refusal (re): the assistant declines firmly and redirects to a local source\n"
        "  - Math/units: answers must be numerically correct\n\n"
        f"Return ONLY a valid JSON array of exactly {n} objects — no markdown, no commentary:\n"
        '[{"q": "सवाल", "a": "जवाब"}, ...]'
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": cached_block,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            pairs = parse_json_pairs(resp.content[0].text)
            cache_read = getattr(resp.usage, "cache_read_input_tokens", 0)
            cache_created = getattr(resp.usage, "cache_creation_input_tokens", 0)
            print(f"    tokens — in:{resp.usage.input_tokens} out:{resp.usage.output_tokens} "
                  f"cache_read:{cache_read} cache_created:{cache_created}")
            return pairs
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    parse error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)
        except Exception as e:
            print(f"    API error (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5)
    return []


def main(dry_run: bool = False):
    client = None
    if not dry_run:
        try:
            import anthropic as ant
        except ImportError:
            sys.exit("anthropic not installed. Run: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            try:
                from google.colab import userdata
                api_key = userdata.get("ANTHROPIC_API_KEY")
            except Exception:
                pass

        client = ant.Anthropic(api_key=api_key) if api_key else ant.Anthropic()

    records: list[dict] = []

    # --- Step 1: write all 20 gold examples ---
    for _cat, q, a in GOLD_EXAMPLES:
        records.append(make_training_record(q, a))
    print(f"Gold examples written: {len(GOLD_EXAMPLES)}")

    if dry_run:
        print("--dry-run: skipping API generation.")
    else:
        cached_block = build_cached_generation_prompt()
        total_generated = 0

        for cat, desc, target_n in CATEGORIES:
            print(f"\n[{cat}] {desc[:60]}...")
            remaining = target_n
            cat_total = 0

            while remaining > 0:
                batch_n = min(BATCH_SIZE, remaining)
                print(f"  requesting {batch_n} examples (remaining {remaining})...")
                try:
                    pairs = generate_batch(client, cached_block, cat, desc, batch_n)
                except Exception as e:
                    print(f"  FAILED after retries: {e}  — skipping remainder of [{cat}]")
                    break

                for q, a in pairs:
                    records.append(make_training_record(q, a))
                cat_total += len(pairs)
                remaining -= len(pairs)
                total_generated += len(pairs)
                print(f"  [{cat}] {cat_total}/{target_n} done")

                if remaining > 0:
                    time.sleep(0.5)  # gentle rate-limit courtesy

        print(f"\nGenerated: {total_generated} new examples")

    # --- Step 2: write JSONL ---
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(records)} examples -> {OUTPUT_FILE}")
    print("Ready for TRL SFTTrainer QLoRA fine-tuning.")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
