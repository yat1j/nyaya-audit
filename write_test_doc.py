"""write_test_doc.py — run once to create a test Firestore document"""
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Write test document with fake but realistic data
db.collection('audit_jobs').document('test-001').set({
    'status':                    'complete',
    'model_name':                'IndicBERT-based hiring model',
    'caste_seat_before':         0.74,
    'caste_seat_after':          0.09,
    'caste_interpretation_before': 'large bias — severe',
    'caste_interpretation_after':  'no significant bias',
    'religion_seat_before':      0.43,
    'religion_seat_after':       0.08,
    'demographic_parity_before': 0.41,
    'demographic_parity_after':  0.87,
    'passes_fairness':           True,
    'shortlist_rates_before':    {'brahmin': 72.0, 'dalit': 29.0},
    'shortlist_rates_after':     {'brahmin': 54.0, 'dalit': 47.0},
    'gemini_explanation':        'The IndicBERT-based model showed severe caste bias with a d-score of 0.74, meaning Brahmin-surname applicants were geometrically 74% more associated with capability words in embedding space. After Nyaya\'s Hard Debiasing, the d-score dropped to 0.09 and demographic parity improved from 41% to 87%, confirming the model now meets the 80/20 fairness rule.',
    'dataset_rows':              200,
    'created_at':                datetime.now(),
})
print("Test document written to audit_jobs/test-001")
print("Tell Member B to use jobId 'test-001' to test their Results screen.")