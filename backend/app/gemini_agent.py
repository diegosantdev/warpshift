import os
import re
import json

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize Gemini
API_KEY = os.environ.get("GOOGLE_AI_API_KEY")
if API_KEY and genai:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None

def analyze_migration_and_fix(diff: str, cuda_code: str) -> dict:
    """
    Analyzes the HIPIFY diff and original code to find what it missed,
    and writes the necessary fixes using Gemini.
    Returns a dict with 'fixes' (list of dicts) and 'reasoning' (str).
    """
    if not model:
        # Fallback if no API key or package
        return {
            "reasoning": "Gemini API key or package not found. Using fallback heuristics.",
            "fixes": []
        }

    prompt = f"""
    You are an expert AMD ROCm engineer migrating CUDA code.
    HIPIFY converted this CUDA code to HIP, but HIPIFY only does simple string replacements.
    It often misses contextual issues like:
    - warpSize hardcoded to 32 (MI300X uses wavefront 64, so it should use __AMDGCN_WAVEFRONT_SIZE or hipWarpSize)
    - cuBLAS/rocBLAS argument order mismatch
    - Hardcoded threading limits

    Analyze this diff and original code. Identify any errors or missing conversions.
    Output MUST be in strict JSON format matching exactly this structure:
    {{
        "reasoning": "Explain your findings briefly here.",
        "fixes": [
            {{
                "issue": "Brief issue title",
                "original_line": "int warpSize = 32;",
                "fixed_line": "int warpSize = hipWarpSize;",
                "line_number": 87,
                "level": "high"
            }}
        ]
    }}

    DIFF:
    {diff}

    ORIGINAL CUDA CODE (snippet):
    {cuda_code[:2000]}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text
        # Extract JSON
        json_match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
        return json.loads(text)
    except Exception as e:
        print(f"Gemini API Error: {e}", flush=True)
        return {"reasoning": f"Error calling Gemini: {e}", "fixes": []}


def generate_pr_body(fixes: list, diff: str) -> str:
    """
    Generates a Pull Request body with reasoning using Gemini.
    """
    if not model:
        return "## Summary\nMigration completed using fallback rules."

    prompt = f"""
    You are the WarpShift AI Agent. You just successfully migrated a CUDA repository to ROCm for an AMD MI300X GPU.
    Write a highly technical, professional GitHub Pull Request description explaining what you did.
    
    Here are the specific fixes you identified and applied on top of the base HIPIFY:
    {json.dumps(fixes, indent=2)}

    Format it cleanly in Markdown. Include a "## Agent Reasoning" section explaining WHY these fixes were made. Do not output anything outside of the markdown itself.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.replace("```markdown", "").replace("```", "").strip()
    except Exception as e:
        return f"Error generating PR body: {e}"
