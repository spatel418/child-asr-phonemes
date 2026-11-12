"""Build the SPROUT v2 manifest (one row per clip) from the local fixed cut.

Source (read-only): /projects/p32576/simron/scratch/sprout_merge/staging_legacy/
Labels: sprout_merge/participants.tsv (at_risk_decision 1=lang 2=speech 3=both 4=not 5=undet)
Sealed SIP test children: /projects/p32576/simron/sip_v2/protocol/test_ids.txt
Audio is not converted here; paths point at the 48 kHz source wavs.
"""
import csv, glob, json, os, sys

SRC = "/projects/p32576/simron/scratch/sprout_merge/staging_legacy"
PART = "/projects/p32576/simron/scratch/sprout_merge/participants.tsv"
SEALED = "/projects/p32576/simron/sip_v2/protocol/test_ids.txt"
OUT = sys.argv[1]

GROUP = {"1": "language-only", "2": "SSD", "3": "SSD", "4": "typical", "5": "unknown"}
part = {r["participant_id"]: r for r in csv.DictReader(open(PART), delimiter="\t")}
sealed = {l.strip() for l in open(SEALED) if l.strip()}

rows = 0
with open(OUT, "w") as out:
    for js in sorted(glob.glob(f"{SRC}/sub-*/beh/*desc-utt*.json")):
        d = json.load(open(js))
        cid = js.split("/")[-3].replace("sub-", "")
        p = part.get(cid, {})
        risk = (p.get("at_risk_decision") or "").split(".")[0]
        age_m = p.get("ss_child_age") or ""
        row = {
            "path": js[:-5] + ".wav",
            "corpus": "SPROUT",
            "child_id": f"SPROUT_{cid}",
            "task": js.split("_task-")[1].split("_")[0],
            "duration": d.get("audio_duration_sec"),
            "age_months": float(age_m) if age_m else None,
            "sex": {"1.0": "M", "2.0": "F"}.get(p.get("ss_child_biosex", ""), None),
            "group": GROUP.get(risk, "unknown"),
            "at_risk_decision": risk or None,
            "site": p.get("data_collection_site") or None,
            "orthographic": d.get("orthographic_text_normalized"),
            "ipa_original": d.get("phonetic_text"),
            "ipa_dd_normalized": d.get("phonetic_text_normalized"),
            "tier": d.get("tier"),
            "should_scrub": d.get("should_scrub"),
            "sip_sealed": cid in sealed,
            "source_sr": d.get("source_sample_rate"),
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows += 1
print("rows", rows)
