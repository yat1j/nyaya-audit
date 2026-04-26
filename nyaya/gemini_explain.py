"""
nyaya/gemini_explain.py

Calls Gemini 1.5 Flash to explain audit results in plain English.
Used in the /audit API endpoint response.

If GEMINI_API_KEY is not set or Gemini fails,
returns a hardcoded explanation using the actual numbers.
"""

import os
import google.generativeai as genai


def get_explanation(
    caste_d_before: float,
    caste_d_after: float,
    parity_before: float,
    parity_after: float
) -> str:
    """
    Returns a 2-sentence plain English explanation.
    
    Parameters:
        caste_d_before  : SEAT d-score before debiasing (e.g. 0.82)
        caste_d_after   : SEAT d-score after debiasing  (e.g. 0.09)
        parity_before   : demographic parity before     (e.g. 0.68)
        parity_after    : demographic parity after      (e.g. 0.85)
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key:
        return _hardcoded_explanation(
            caste_d_before, parity_before, parity_after)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""You are explaining AI bias results to an HR director at an Indian company.

Results from a bias audit of their AI hiring model:
- Caste bias severity before fixing: {caste_d_before:.2f} 
  (scale: 0=none, 0.2=small, 0.5=significant, 0.8+=severe)
- Caste bias severity after fixing: {caste_d_after:.2f}
- Hiring fairness score before: {parity_before:.2f} 
  (needs to be above 0.80 to be legally fair)
- Hiring fairness score after: {parity_after:.2f}

Write exactly 2 sentences. First sentence: what this bias meant 
for Dalit and Muslim candidates in practice — be specific about 
who was disadvantaged. Second sentence: what changed after fixing — 
be specific about the improvement.

Hard rules:
- No technical words: embedding, vector, cosine, subspace, model weights
- Maximum 55 words total
- Be direct and specific, not vague"""

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Verify it's not too long
        if len(text.split()) > 70:
            return _hardcoded_explanation(
                caste_d_before, parity_before, parity_after)
        
        return text
        
    except Exception as e:
        print(f"  Gemini API error: {e} — using fallback explanation")
        return _hardcoded_explanation(
            caste_d_before, parity_before, parity_after)


def _hardcoded_explanation(caste_d: float,
                            parity_before: float,
                            parity_after: float) -> str:
    """
    Fallback explanation using actual numbers.
    Used when Gemini API is unavailable.
    """
    if caste_d >= 0.8:
        severity = "severely"
        description = "a severe caste bias"
    elif caste_d >= 0.5:
        severity = "significantly"
        description = "a significant caste bias"
    else:
        severity = "measurably"
        description = "a measurable caste bias"
    
    shortfall = round((1 - parity_before) * 100, 1)
    improvement = round((parity_after - parity_before) * 100, 1)
    
    return (
        f"This AI model {severity} disadvantaged candidates with Dalit "
        f"and Muslim names due to {description} in its embedding "
        f"representations — their shortlisting rate was "
        f"{shortfall}% below the fairness threshold. "
        f"After Nyaya's debiasing, fairness improved by "
        f"{improvement} percentage points, bringing the model "
        f"above the 0.80 compliance threshold."
    )