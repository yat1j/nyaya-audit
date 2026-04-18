import pickle

print("Loading scoring_config.pkl...")
config = pickle.load(open('data/scoring_config.pkl', 'rb'))

print("Keys in config:", list(config.keys()))
print()

# Save each component as a separate pkl file
# Day 3 FastAPI loads these individually at startup

pickle.dump(config['caste_subspace'],
            open('data/caste_dir.pkl', 'wb'))
print("Saved caste_dir.pkl — shape:", config['caste_subspace'].shape)

pickle.dump(config['religion_subspace'],
            open('data/religion_dir.pkl', 'wb'))
print("Saved religion_dir.pkl — shape:", config['religion_subspace'].shape)

pickle.dump(config['cap_profile_biased'],
            open('data/cap_profile_biased.pkl', 'wb'))
print("Saved cap_profile_biased.pkl")

pickle.dump(config['cap_profile_debiased'],
            open('data/cap_profile_debiased.pkl', 'wb'))
print("Saved cap_profile_debiased.pkl")

pickle.dump(config['lim_profile_biased'],
            open('data/lim_profile_biased.pkl', 'wb'))
print("Saved lim_profile_biased.pkl")

pickle.dump(config['lim_profile_debiased'],
            open('data/lim_profile_debiased.pkl', 'wb'))
print("Saved lim_profile_debiased.pkl")

print()
print("All files saved. Verifying...")

# Verify all files exist and load correctly
import os
files_to_check = [
    'data/caste_dir.pkl',
    'data/religion_dir.pkl',
    'data/cap_profile_biased.pkl',
    'data/cap_profile_debiased.pkl',
    'data/lim_profile_biased.pkl',
    'data/lim_profile_debiased.pkl',
    'data/scoring_config.pkl',
    'data/classifier_results.json',
    'data/seat_results.json',
    'data/word_lists.json',
    'data/demo_dataset.csv'
]

print()
print("File check:")
all_good = True
for f in files_to_check:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    status = "OK" if exists else "MISSING"
    print(f"  {status}  {f}  ({size} bytes)")
    if not exists:
        all_good = False

print()
if all_good:
    print("Day 2 complete. All files present.")
    print("You are ready for Day 3.")
else:
    print("Some files are missing. Check errors above.")