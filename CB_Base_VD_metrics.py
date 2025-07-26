import os
import time
import json
import torch, gc
from typing import Any, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Free up memory at the start
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

model_id = "mrm8488/codebert-base-finetuned-detect-insecure-code"
access_token = 'XXX'

MODEL_MAX_LENGTH = 200000

def extract_cve_info(filename):
    """
    Extract CVE ID and description from a JSON file.

    Args:
        filename: Path to JSON file containing the dataset

    Returns:
        tuple: (cve_id, description) both as strings if single entry
        list of tuples: [(cve_id, description), ...] if multiple entries
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # Check if data is a list (multiple entries) or single entry
    if isinstance(data, list):
        results = []
        for entry in data:
            cve_id, description = process_entry(entry)
            results.append((cve_id, description))
        return results
    else:
        # Single entry
        return process_entry(data)


def process_entry(data):
    """
    Process a single dataset entry to extract CVE information.

    Args:
        data: Dictionary containing a dataset entry

    Returns:
        tuple: (cve_id, description) both as strings
    """
    # Extract CVE ID (already a string)
    cve_id = data.get('cve_id', '')

    # Extract description
    description_list = data.get('description', [])

    # If description is a list of dictionaries, extract the 'value' field
    if isinstance(description_list, list) and len(description_list) > 0:
        # Get the first description entry (assuming English)
        desc_entry = description_list[0]
        if isinstance(desc_entry, dict):
            description = desc_entry.get('value', '')
        else:
            description = str(description_list)
    else:
        description = str(description_list)

    return cve_id, description


# Example usage:
# For a single entry JSON file
cve_result = extract_cve_info('cve_output.json')

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=MODEL_MAX_LENGTH,
        truncation=True,
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_model(model_id):
    # Use 4-bit quantization to reduce memory footprint
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir='/home/neeraj/experiment/install_dir/.cache/huggingface/huggingface'
    )
    print(f"Model loaded on device: {model.device}")
    return model

def get_pipe(model, tokenizer):
    torch.cuda.empty_cache()
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        truncation=True,
        max_new_tokens=1024,
        temperature=1,
        do_sample=True,
        device_map="auto",
        use_cache=True,
    )

class DetectionAgent:
    def __init__(
        self,
        *,
        model: str = None,
        temperature: float = 1,
        max_code_lines: int = 400,
        max_ast_chars: int = 51_200,
        max_ctx_chars: int = 11_264,
        pipe=None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_code_lines = max_code_lines
        self.max_ast_chars = max_ast_chars
        self.max_ctx_chars = max_ctx_chars
        self.pipe = pipe

    def __call__(
        self,
        *,
        clean_code: str,
        compact_ast: Dict[str, Any]
    ) -> Dict[str, Any]:
        messages = self._build_messages(
            clean_code, compact_ast
        )
        raw = self._invoke_llm(messages)
        return raw

    # ───────── prompt building ────────
    """
    Tested the below prompt and got this result: 
    Accuracy: 0.6000
    Precision: 0.0000
    Recall: 0.0000
    F1 Score: 0.0000
    """
    # _SYS = (
    #     "You are a C++ security expert responsible for detecting vulnerabilities in code. "
    #     "Examine the given function to identify if it has any security weaknesses, "
    #     "without including details on how to exploit them. "
    #     "Only label the function as vulnerable if you are entirely confident. "
    #     "When a vulnerability is detected, indicate the precise type of vulnerability. "
    #     "Avoid adding any extra details or explanations beyond the type. "
    #     "Follow the output schema provided below exactly and do not include additional content:\n"
    #     '{\n'
    #     '  "is_vulnerable": <true or false>,\n'
    #     '  "vulnerability": "<specific vulnerability type or empty string if not vulnerable>"\n'
    #     '}\n'
    # )
    """
    Tested the below prompt and got this result: 
    Accuracy: 0.6200
    Precision: 0.5185
    Recall: 0.7000
    F1 Score: 0.5957
    """
    _SYS = (
        "You are a C++ security expert with a singular focus on detecting vulnerabilities in code. "
        "Your only task is to analyze the provided function for security flaws, without offering any exploit details or unrelated commentary. "
        "Classify the function as vulnerable only if you have absolute certainty, based on concrete evidence in the code. "
        "If a vulnerability is identified, state only the exact type of vulnerability, nothing more. "
        "Under no circumstances should you provide explanations, reasoning, or additional information beyond what is specified. "
        "You must follow the exact output schema below, with no deviations or extra content allowed:\n"
        '{\n'
        '  "vulnerable": <true or false>,\n'
        '  "vulnerability": "<specific vulnerability type or empty string if not vulnerable>"\n'
        '}\n'
        "Failure to adhere to this format or inclusion of unsolicited text will be considered incorrect. Respond strictly within these constraints."
    )

    """
    Tested the below prompt and got this result: 
    Accuracy: 0.6000
    Precision: 0.0000
    Recall: 0.0000
    F1 Score: 0.0000
    """
    # _SYS = (
    #     "You are a C++ security expert assigned exclusively to detect vulnerabilities in provided code. "
    #     "Your sole responsibility is to evaluate the given function for security flaws. Do not consider any other aspect or provide exploit methods. "
    #     "Classify the function as vulnerable only if you are 100% certain, supported by undeniable evidence in the code itself. "
    #     "If a vulnerability exists, report only the precise type of vulnerability, without any elaboration, description, or context. "
    #     "Do not under any condition include opinions, explanations, justifications, or content beyond the specified fields. "
    #     "Your output must conform exactly to the schema below, with zero tolerance for deviations, additional text, or formatting errors:\n"
    #     '{\n'
    #     '  "is_vulnerable": <true or false>,\n'
    #     '  "vulnerability": "<specific vulnerability type or empty string if not vulnerable>"\n'
    #     '}\n'
    #     "Any response that violates this format, includes extra content, or fails to follow instructions will be deemed invalid. "
    #     "Adhere to these rules without exception and respond only with the required JSON structure."
    # )

    """
    Tested the below prompt and got this result: 
    Accuracy: 0.6000
    Precision: 0.0000
    Recall: 0.0000
    F1 Score: 0.0000
    """
    # _SYS = (
    #     "ROLE: C++ vulnerability detection system.\n"
    #     "TASK: Analyze the provided C++ function for security vulnerabilities ONLY.\n"
    #     "RULES:\n"
    #     "1. Output EXACTLY this JSON format - no variations:\n"
    #     '   {"is_vulnerable": <boolean>, "vulnerability": "<type>"}\n'
    #     "2. Set is_vulnerable=true ONLY with absolute certainty and direct code evidence.\n"
    #     "3. If vulnerable, vulnerability field must contain ONLY the vulnerability type name.\n"
    #     "4. If not vulnerable, vulnerability field must be an empty string \"\".\n"
    #     "5. FORBIDDEN: explanations, comments, markdown, extra whitespace, or any text outside the JSON.\n"
    #     "6. FORBIDDEN: exploit details, remediation advice, or severity assessments.\n"
    #     "7. Valid vulnerability types: buffer overflow, integer overflow, use after free, double free, format string, command injection, SQL injection, path traversal, race condition, null pointer dereference.\n"
    #     "8. Response must be valid JSON - boolean without quotes, string with quotes.\n"
    #     "9. Any deviation from this format invalidates the response.\n"
    #     "RESPOND WITH JSON ONLY."
    # )

    def _build_messages(
        self,
        code: str,
        ast: Dict[str, Any]
    ):
        numbered = self._number_lines(code)
        numbered = self._truncate(numbered, 5000)  # Limit input size
        ast_str = self._truncate(json.dumps(ast, separators=(",", ":")), self.max_ast_chars)
        user = (
            "### Function (line-numbered):\n"
            f"{numbered}\n\n"
            "### Compact AST (could be truncated if very long):\n"
            f"{ast_str}\n\n"
        )
        return [
            {"role": "system", "content": self._SYS},
            {"role": "system", "content": f"Read through this file and understand all the CVE and its description: \n\n{cve_result}"},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _number_lines(code: str) -> str:
        return "\n".join(f"{i:4d}: {line}" for i, line in enumerate(code.splitlines()))

    @staticmethod
    def _truncate(txt: str, limit: int) -> str:
        return txt if len(txt) <= limit else txt[: limit] + " …[truncated]…"

    def _invoke_llm(self, messages):
        BATCH_SIZE = 1
        # print(f"Memory allocated before inference: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
        # print(f"Memory reserved before inference: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GiB")
        with torch.no_grad():  # Disable gradient computation
            resp = self.pipe(messages, temperature=self.temperature, do_sample=True, batch_size=BATCH_SIZE)
        # Debug the structure of the response to ensure correct extraction
        # print("Response structure (first 200 chars of raw output):", str(resp[0]["generated_text"])[:200] + "...")
        response_text = resp[0]["generated_text"][-1]['content'].strip()
        # print("Extracted response text (first 200 chars):", response_text[:200] + "...")
        return response_text


def parse_response(response_text):
    """
    Extract 'is_vulnerable' boolean and optional 'vulnerability' description from the model's response text.
    Returns a tuple of (is_vulnerable: bool, vulnerability: str or None).
    """
    try:
        if not isinstance(response_text, str):
            print(f"Error: response_text is not a string, it's a {type(response_text)}")
            return False, None

        vulnerable = False
        vulnerability = None

        # Check for is_vulnerable
        if '"vulnerable": true' in response_text.lower():
            vulnerable = True
        elif '"vulnerable": false' in response_text.lower():
            vulnerable = False
        else:
            match = re.search(r'"vulnerable":\s*(true|false)', response_text, re.IGNORECASE)
            if match:
                vulnerable = match.group(1).lower() == 'true'

        # Check for vulnerability description
        vuln_match = re.search(r'"vulnerability":\s*"([^"]+)"', response_text)
        if vuln_match:
            vulnerability = vuln_match.group(1)

        return vulnerable, vulnerability
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(
            f"Original State: Response text: {response_text[:200] if isinstance(response_text, str) else response_text}...")
        return False, None


def process_data(data):
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    vulnerabilities = []  # Store vulnerability descriptions if available

    for item in data:
        det_inputs = dict(
            clean_code=item["clean_code"],
            compact_ast=item["compact_ast"]["ast"]
        )
        true_label = item.get("vulnerable", False)
        y_true.append(true_label)

        response_text = det(**det_inputs)
        predicted_label, vuln_desc = parse_response(response_text)
        y_pred.append(predicted_label)
        vulnerabilities.append(vuln_desc if vuln_desc else "N/A")

        # print(f"Ground Truth: {true_label}, Predicted: {predicted_label}, Vulnerability: {vuln_desc if vuln_desc else 'N/A'}")
        # print('------------------------------------------------------------------------------------------')
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"Memory allocated after inference: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB")
        # print(f"Memory reserved after inference: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GiB")

    # Compute metrics
    if y_true and y_pred:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print("Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    else:
        print("No data to evaluate metrics.")

pipe = get_pipe(
    get_model(model_id),
    get_tokenizer(model_id)
)

det = DetectionAgent(model=model_id, pipe=pipe)

# def process_data(data):
#     for item in data:
#         # print(f"The model is working on the {item}")
#         det_inputs = dict(
#             clean_code=item["clean_code"],
#             compact_ast=item["compact_ast"]["ast"]
#         )
#         detection = det(**det_inputs)
#         print('------------------------------------------------------------------------------------------')
#         gc.collect()
#         torch.cuda.empty_cache()
#         # print(f"Memory allocated after inference: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
#         # print(f"Memory reserved after inference: {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

with open('result_output.json', "r") as f:
# with open('new_result_output.json', "r") as f:
    data = json.load(f)
process_data(data)
