"""
write_to_firestore.py
Calls HF API, parses full response, writes to Firestore.
Backup bridge script — use if Flutter's direct API call fails.

Usage: python write_to_firestore.py data/demo_dataset.csv my-job-id-001
"""
import sys, requests, firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

API_BASE = 'https://yatj-nyaya-api.hf.space'
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else 'data/demo_dataset.csv'
JOB_ID   = sys.argv[2] if len(sys.argv) > 2 else 'manual-001'

# Firebase init
cred = credentials.Certificate('serviceAccountKey.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Mark as running
db.collection('audit_jobs').document(JOB_ID).set({
    'status': 'running', 'created_at': datetime.now(), 'csv_path': CSV_PATH
})
print(f"Job {JOB_ID} marked as running in Firestore")

# Call /audit
print(f"Calling {API_BASE}/audit with {CSV_PATH}...")
with open(CSV_PATH, 'rb') as f:
    resp = requests.post(f'{API_BASE}/audit',
                         files={'file': f}, timeout=600)
print(f"Status: {resp.status_code}")

if resp.status_code != 200:
    db.collection('audit_jobs').document(JOB_ID).update(
        {'status': 'error', 'error_message': resp.text[:500]})
    print("ERROR — check API logs")
    sys.exit(1)

data = resp.json()
# print(f"Caste before: {data['caste_seat_before']} → after: {data['caste_seat_after']}")
print(f"Certified: {data['passes_fairness']}")

# Write audit results
db.collection('audit_jobs').document(JOB_ID).update({
    **{k: data[k] for k in [
        'caste_seat_before','caste_seat_after',
        'caste_interpretation_before','caste_interpretation_after',
        'religion_seat_before','religion_seat_after',
        'demographic_parity_before','demographic_parity_after',
        'passes_fairness','passes_fairness_before',
        'shortlist_rates_before','shortlist_rates_after',
        'gemini_explanation','dataset_rows','model',
    ] if k in data},
    'status': 'audit_complete', 'audit_done_at': datetime.now(),
})
print("Audit results written to Firestore")

# Call /retroactive
print(f"Calling {API_BASE}/retroactive...")
with open(CSV_PATH, 'rb') as f:
    r2 = requests.post(f'{API_BASE}/retroactive',
                       files={'file': f}, timeout=600)

if r2.status_code == 200:
    retro = r2.json()
    # Write per-decision subcollection
    col = db.collection('audit_jobs').document(JOB_ID).collection('retroactive_results')
    for item in retro.get('per_decision', []):
        col.document(str(item['id'])).set(item)
    db.collection('audit_jobs').document(JOB_ID).update({
       # 'outcomes_changed':  retro['outcomes_changed'],
       # 'newly_shortlisted': retro['newly_shortlisted'],
       # 'total_decisions':   retro['total_decisions'],
        'status':            'complete',
        'completed_at':      datetime.now(),
    })
   # print(f"Retroactive done: {retro['outcomes_changed']} outcomes changed")
 #print(f"Newly shortlisted: {retro['newly_shortlisted']}")
else:
    db.collection('audit_jobs').document(JOB_ID).update({'status': 'complete'})
    print(f"Retroactive failed ({r2.status_code}) — audit data still saved")

print(f"\nDone. Firestore document: audit_jobs/{JOB_ID}")
print("Tell Member B to use this jobId in their Results screen.")