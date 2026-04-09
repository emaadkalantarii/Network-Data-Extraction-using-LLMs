"""
High‑level flow
===============
1. First check if the resolved_text.txt file exists at the same directory or not.
2. If it exists, prompt the user to review it, and press Enter to continue with the file, and enter the "new" if the user wants to run from scratch with the input text (ignore the file), and the same thing if the file does not exist (start from scratch)
3. Clean raw text (trim whitespace & newlines) and save the cleaned text in a txt file.
4. Split cleaned text into overlapping chunks and save the chunks in a txt file.
5. Logging the numbers of chunks and prompt the user to enter the chunk numbers to be proceeded.
6. For each selected chunks, the first stage ( resolving the pronouns and family names from the chosen chunk text with providing a context window or previous chunk/chunks text) will be processed and save the processed text for all the chosen chunks in a txt file.
7. For the second stage (NER or Relationship Extraction), the processed text for each chunks will be used and asked by a prompt.
8. The results from all the chunks will be saved in a json file with a specific structure. 
Note:
The model used for the first and second stage could be chosen from Reasoning Models and GPT models with their related parameters ( either using reasoning effort parameter for Reasoning Models or Top_p, Temperature, and max_tokens parameters for GPT Models).
"""

import ast
import json
import os
import re
from datetime import datetime

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter


client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 

#  Model Selection:
#Stage1 : CR -> Coreference Resolution : Resolving the pronouns and family names to their actual names.
STAGE1_MODEL = "o3-2025-04-16"  # e.g. "gpt-4.0", "o3-2025-04-16", "gpt-4-1106-preview", "gpt-5-mini-2025-08-07", "gpt-5-2025-08-07", etc.

#Stage2: Data extraction -> Identifying entities with their relationships (Named Entity Recognition + Relationship Extraction).
STAGE2_MODEL = "o3-2025-04-16"    # e.g. "gpt-4.1", "o3", "gpt-4.1-2025-04-14", "gpt-5-nano-2025-08-07", etc.

# Optional model params - used only for GPT models
STAGE1_MODEL_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.3,
    "max_tokens": 32768,
}
STAGE2_MODEL_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.3,
    "max_tokens": 32768,
}

# Optional model params - used only for Reasoning models
STAGE1_REASONING_EFFORT = "high"
STAGE2_REASONING_EFFORT = "high"


# Optional model params - used only for GPT-5 models
STAGE1_GPT5_PARAMS = {
    "verbosity": "high",
    "reasoning_effort": "high",
    "max_output_tokens": 32768,
}
STAGE2_GPT5_PARAMS = {
    "verbosity": "high",
    "reasoning_effort": "high",
    "max_output_tokens": 32768,
}

# Text Splitting Configurations (chunking)
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 600

# Context window size for determing how many previous chunks should be given as the reference in CR process for resolving pronouns and family names.
CONTEXT_WINDOW = 1

# After CR process, the Resolved text (Including the processed version of the chosen chunks) will be saved in this filename. 
RESOLVED_TXT_FILENAME = "resolved_text.txt"

# After cleaning and chunking steps, the related text will be saved also in the corresponding files to let the user to inspect the files.


# Content
input_text = """insert your source text here """

#Stage1 (CR) system and user prompts:
pronoun_resolution_system_message   = """
insert your instruction, general rules, pronoun and family resolution rules text here
e.g.

# INSTRUCTION:
You are a text pre-processor specializing in reference resolution.
Your task is to identify pronouns and family-name-only references, and append the corresponding entity names in square brackets [].

## GENERAL RULES (apply to both)

- Do ​not​ delete, re-order, or paraphrase any original words, punctuation, or line breaks.
- Preserve the original text structure and punctuation
- Insert clarifications by adding exactly one space after the target word/phrase, then the clarification in square brackets […].
- Return only the modified text, without any explanations or additional information, no headings, no extra lines (do not add new lines (\\n)).


## Pronoun Resolving Rules:

1. Identify all pronouns (he, she, it, they, we, us, them, him, her, their, our, etc.)
2. Find the entities these pronouns refer to
3. Keep pronouns in their original position and add [entity names] immediately after each pronoun
4. If a pronoun refers to multiple entities, list all of them: pronoun [entity1, entity2, entity3]


## Family Name or Surname Resolving Rules:

1. Identify family-name-only references: Forms like “the <FamilyName>” or a bare surname used to denote a household or lineage.
2. Treat the reference as the adult head(s) of the household. 
3. After each family-name-only reference, If the parents’ names of that family, or adult heads of that household names appear elsewhere in the text, add square brackets containing the entities and list them: family name [Parent 1 name, Parent 2 name]. (e.g., “the Johns [Alice John, Bob John]”. 
4. After each family-name-only reference, If the parents’ names of that family, or adult heads of that household names are not mentioned in the text, leave the family name alone: family name [family name] (e.g., “the Johns [the Johns]”).

"""
pronoun_resolution_prompt_template  = """You will be given text that may contain previous context followed by the main text to process. Add entity references after all pronouns and family-name-only references in the MAIN TEXT ONLY. Keep each reference in its original position and add the entities/entity it refers to in square brackets [] immediately after the pronoun and family-name-only references.

IMPORTANT: Only modify the MAIN TEXT section. The CONTEXT sections are provided solely to help you identify what each reference refers to.

{context_section}

MAIN TEXT TO PROCESS:
{main_text}

Please return ONLY the processed version of the MAIN TEXT with entity names added after pronouns and family-name-only references."""


#Stage2 (Identifying entities with their relationships) system and user prompts:
ner_system_message  = """
insert your instruction, rules, and definitions text here
e.g.

# INSTRUCTION:
You are a specialized AI assistant for extracting significant, non-routine help-related relationships from historical texts. Your expertise lies in identifying deliberate acts of assistance related to survival, escape, and underground living—particularly those involving risk, sacrifice, or extraordinary effort. Your task is to extract all meaningful help-related relationships from the provided text using ONLY these 9 relationship types:

## RELATIONSHIP TYPES (extract only when representing significant, non-routine assistance):
1. **Providing shelter or protection**: One entity makes a distinct effort to offer or secure a safe place, accommodation, or refuge for another, often in response to a need or to avert danger.
2. **Providing medical care**: One entity actively administers, procures, or arranges healthcare, treatment, or medical aid for another who is sick, injured, or in clear need of such attention.
3. **Providing food or material resources**: One entity intentionally gives, shares, or ensures access to sustenance, essential supplies, money, or other material goods for another, typically to alleviate a need or hardship.
4. **Making introductions or connections**: This involves a 'Facilitator' entity who purposefully acts to establish a connection or introduction between a 'Beneficiary' (who needs help) and a 'Potential Helper', aiming for a positive outcome for the Beneficiary.
   - If such a facilitation occurs, you MUST identify and list separately in the JSON output:
     - a) The connection relationship: `entity1: Facilitator`, `entity2: Beneficiary`, `Form of help: Making introductions or connections`.
     - b) The connection relationship: `entity1: Facilitator`, `entity2: Potential Helper`, `Form of help: Making introductions or connections`.
     - c) The other relationship: `entity1: Potential Helper`, `entity2: Beneficiary`, `Form of help: relationship under one of the other 8 types`. (Any subsequent actual help provided by the Potential Helper to the Beneficiary should be extracted as a separate relationship under one of the other 9 types.)
5. **Providing employment or work opportunities**: One entity specifically offers, creates, or secures a job, position, or a means of earning income for another.
6. **Providing false documentation or identity assistance**: One entity deliberately acts to help another obtain or use forged documents or a false identity, usually to overcome a significant obstacle or ensure safety.
7. **Sharing information or advice**: One entity consciously imparts specific, actionable knowledge, guidance, or counsel to another, with the evident intention of being helpful or beneficial to the recipient's situation.
8. **Providing emotional support or companionship**: One entity makes a clear and supportive effort to offer comfort, encouragement, motivation, friendship, or psychological aid to another, particularly during a challenging time or in response to emotional distress.
9. **Other helping or assisting**: One entity performs a distinct, tangible action to aid another in a significant way not covered by the types above, such as direct physical assistance in a task, critical translation services, necessary transportation in a specific situation, or other clear favors that provide a demonstrable benefit.

## DETAILED RELATIONSHIP DEFINITIONS:

### Providing shelter or protection
Encompasses any arrangement where physical refuge is secured, from permanent accommodation in homes to temporary hideouts in attics, cellars, or institutions. Include cases where accommodation is arranged through intermediaries, even if the arranger doesn't directly host. Emergency shelters independently found are self-help, not coded relationships.

### Providing medical care
Covers all healthcare-related assistance, from direct treatment by medical professionals to procurement of medicines or arrangement of medical consultations. The focus is on deliberate acts to address health needs during persecution.

### Providing food or material resources
Includes direct food provision, sharing ration cards, facilitating black market access, or providing money/supplies for sustenance. The relationship exists between the provider and recipient, even when resources pass through intermediaries. Self-procurement (e.g., eating in restaurants) is not coded.

### Making introductions or connections
A unique triangular relationship where mediation creates new helping opportunities. The facilitator's act of connecting must be coded as two relationships (facilitator→beneficiary and facilitator→potential helper), plus any subsequent help as a separate relationship. This captures how networks expanded through personal connections.

### Providing employment or work opportunities
Covers both genuine employment and fictitious work arrangements created for protection. Include cases where employment provides cover identity or legitimacy, not just income.

### Providing false documentation or identity assistance
Encompasses the entire chain of document creation, procurement, and delivery. Include air raid damage certificates and any papers that helped establish new identities. The relationship exists even when documents pass through multiple hands before reaching the final recipient.

### Sharing information or advice
Focuses on actionable intelligence or guidance that directly impacts survival or safety. Must be specific information intended to help, not casual conversation. Warnings about raids, advice on safe routes, or instructions for obtaining resources all qualify.

### Providing emotional support or companionship
Recognizes the psychological dimension of help during persecution. Include visits to those in hiding, acts breaking social segregation, gestures affirming human dignity, and ongoing relationships that provided moral support. Context often reveals the significance of seemingly simple acts.

### Other helping or assisting
Captures significant assistance not covered above, such as critical translations, essential transportation, or other tangible favors with clear beneficial impact. Must represent deliberate, meaningful help beyond routine interactions.


## Extraction Rules:
- "entity1" MUST be the entity providing the help/support full name
- "entity2" MUST be the entity receiving the help/support full name
- "Form of help" MUST be one of the 9 exact phrases listed above
- "evidence" MUST be an exact, continuous text snippet that shows the helping action
- If the same relationship (with the same "entity1", "entity2" and "Form of Help" but different in context "evidence") appears multiple times, list each occurrence separately with its specific evidence
- If entity1 or entity2 is a pronoun (we, us, they, them, you, etc.), use only the entity names inside the brackets as the entity. prounoun [entity names] (e.g., they [students, teacher]). If no brackets appear, or the brackets contain no entity names, use the original pronoun as the entity.
- If entity1 or entity2 is a Family name (the <FamilyName>, 'the Family name', 'the Johns', 'the Lorens', etc.), treat the provided individual names inside the brackets as the actual entities—NOT the family name itself. Use only the entity names provided within the brackets as the entity (Use ONLY the names I've explicitly provided in the brackets.). Do not infer or reason about who else might belong to this family based on the text. the Family name [Parent 1 name and Family name, Parent 2 name and Family name]  (e.g., the Johns [Alex John, Alice John]). If no brackets appear after a family name, or if the brackets are empty, then use the family name as the entity.
- If entity1 or entity2 is a list of entities/entity, do not list relationship for each entity separately, having one relationship for all the listed entities is enough as all of the listed entities are related to either entity1 or entity2.

## Historical Context Focus:
This extraction focuses on relationships during World War II where help was provided for survival, escape, hiding, or resistance. Prioritize:
1. Help provided to those in hiding or fleeing persecution
2. Assistance that risked the helper's own safety or life
3. Underground networks and resistance support
4. Acts that enabled survival during occupation or persecution
5. Help that violated official regulations to save lives
6. Cascading help where one form of assistance enabled access to other resources

## Exception – Critical bureaucratic assistance
Sometimes a routine, lawful service provided by a government office, charity, or other institution nevertheless becomes life-saving for a persecuted person.  Annotate such an act as a help-relationship when ALL of the following are true:
    1- Survival impact The service delivers a benefit that the text links directly to the beneficiary’s safety, ability to hide, escape, or obtain essential resources (e.g., identity papers, visas, ration coupons, protected employment, housing allocation).
    2- Unusual access The beneficiary belongs to a group that would normally be barred from, or put at risk by, the ordinary procedure.
    3- Explicit connection in text The narrative explicitly shows that receiving this service improved the beneficiary’s chances of survival or escape.
Coding note: When these three conditions are met, record the institution (or the official named) as “entity1” and select the appropriate help-type (Providing false documentation, Providing food or material resources, Providing employment or work opportunities, Providing shelter or protection, etc.), even if the officials acted routinely and were unaware of the larger stakes.

## OUTPUT FORMAT:
Return ONLY valid JSON:
{
  "relationships": [
    {
      "entity1": "Helper full name or phrase",
      "entity2": "Recipient full name or phrase",
      "Form of help": "Exact type from list",
      "evidence": "Exact quote from text"
    }
  ]
}

If no relationships found: `{"relationships": []}`.
"""

ner_user_prompt_template = """Extract all help-related relationships from the text below where one entity provides assistance to another for survival, escape, or protection during World War II. Focus only on clear instances of helping actions that represent extraordinary effort, risk, or sacrifice - not routine daily interactions. Consider only relationships that fall into one of the 9 predefined categories in your instructions.
Given the following text:
\"\"\"{chunk}\"\"\""""



# Utility Functions
def is_gpt5_model(model_name: str) -> bool:
    name = model_name.lower().strip()
    return name.startswith("gpt-5")

def is_reasoning_model(model_name: str) -> bool:
    # Accepts o3*, o4, octo etc
    return model_name.lower().startswith("o")

def is_gpt_model(model_name: str) -> bool:
    # Generic GPT models (GPT-4.x, GPT-3.x, 4o, etc.). We treat GPT-5 separately.
    name = model_name.lower()
    if is_gpt5_model(name):
        return False
    return (
        "gpt" in name
        or name.startswith("4o")
        or name.startswith("gpt-4o")
        or name.startswith("gpt-4")
        or name.startswith("gpt-3")
    )

def get_model_log_line(stage_num, model, gpt_params, reasoning_effort, gpt5_params=None):
    """
    gpt_params: dict used for classic GPT models (temperature/top_p/max_tokens)
    reasoning_effort: string for o*-series models
    gpt5_params: dict for GPT-5 models (verbosity/reasoning_effort/max_output_tokens)
    """
    if is_gpt5_model(model):
        v = (gpt5_params or {}).get("verbosity", "default")
        reff = (gpt5_params or {}).get("reasoning_effort", "default")
        mot = (gpt5_params or {}).get("max_output_tokens", "default")
        msg = (
            f"Stage {stage_num} model: {model} (GPT-5 reasoning model)"
            f"\nverbosity: {v}, reasoning_effort: {reff}, max_output_tokens: {mot}"
        )
    elif is_reasoning_model(model):
        msg = f"Stage {stage_num} model: {model} (reasoning model)"
        if reasoning_effort:
            msg += f"\nReasoning effort: {reasoning_effort}"
        else:
            msg += "\nReasoning effort: default"
    elif is_gpt_model(model):
        t = (gpt_params or {}).get('temperature', "default")
        p = (gpt_params or {}).get('top_p', "default")
        mtok = (gpt_params or {}).get('max_tokens', "default")
        msg = f"Stage {stage_num} model: {model} (GPT model)"
        msg += f"\ntemperature: {t}, top_p: {p}, max_tokens: {mtok}"
    else:
        msg = f"Stage {stage_num} model: {model} (custom/unknown)"
    return msg

def get_model_metadata_dict(model, gpt_params, reasoning_effort, gpt5_params=None):
    if is_gpt5_model(model):
        return {
            "model": model,
            "verbosity": (gpt5_params or {}).get("verbosity", None),
            "reasoning_effort": (gpt5_params or {}).get("reasoning_effort", None),
            "max_output_tokens": (gpt5_params or {}).get("max_output_tokens", None),
        }
    elif is_reasoning_model(model):
        return {"model": model, "reasoning_effort": reasoning_effort}
    elif is_gpt_model(model):
        return {
            "model": model,
            "temperature": (gpt_params or {}).get("temperature", None),
            "top_p": (gpt_params or {}).get("top_p", None),
            "max_tokens": (gpt_params or {}).get("max_tokens", None),
        }
    else:
        return {"model": model}

def run_model_call(
    model,
    system_message,
    user_prompt,
    model_params=None,
    reasoning_effort=None,
    gpt5_params=None
):
    """
    Unified model caller:
      - GPT-5 models: Responses API with text/verbosity + reasoning objects
      - o*-series reasoning models: Chat Completions with reasoning_effort
      - Classic GPT models: Chat Completions with temperature/top_p/max_tokens
    """
    # Build chat-style messages for non-GPT-5 paths
    messages = []
    if system_message.strip():
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_prompt})

    if is_gpt5_model(model):
        # Responses API expects a list of role/content messages in `input`
        input_msgs = []
        if system_message.strip():
            input_msgs.append({"role": "system", "content": system_message})
        input_msgs.append({"role": "user", "content": user_prompt})

        params = {
            "model": model,
            "input": input_msgs,
        }

        # text.verbosity must be nested
        text_obj = {}
        if gpt5_params and gpt5_params.get("verbosity") is not None:
            text_obj["verbosity"] = gpt5_params["verbosity"]
        if text_obj:
            params["text"] = text_obj

        # reasoning effort lives under `reasoning`
        eff = (gpt5_params or {}).get("reasoning_effort") or reasoning_effort
        if eff is not None:
            params["reasoning"] = {"effort": eff}

        # optional token cap for GPT-5
        if gpt5_params and gpt5_params.get("max_output_tokens") is not None:
            params["max_output_tokens"] = gpt5_params["max_output_tokens"]

        response = client.responses.create(**params)

        # Extract text robustly across SDK variants
        # Prefer convenience properties, then fall back.
        if hasattr(response, "output_text") and isinstance(response.output_text, str):
            return response.output_text.strip()
        if hasattr(response, "output") and isinstance(response.output, str):
            return response.output.strip()
        # last-resort stringify
        return str(response)

    elif is_reasoning_model(model):
        params = {
            "model": model,
            "messages": messages
        }
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    elif is_gpt_model(model):
        params = {
            "model": model,
            "messages": messages,
        }
        if model_params:
            if model_params.get("temperature") is not None:
                params["temperature"] = model_params["temperature"]
            if model_params.get("top_p") is not None:
                params["top_p"] = model_params["top_p"]
            if model_params.get("max_tokens") is not None:
                params["max_tokens"] = model_params["max_tokens"]
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()

    else:
        # Fallback (treat like classic GPT)
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()


def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = re.sub(r'\.(?=[A-Za-z])', '. ', text)
    return text

def chunk_input(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[".", "!", "?","\n\n", "\n", " ", ""],
        keep_separator='end'
    )
    return splitter.split_text(text)

def save_text_for_inspection(text, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Text saved successfully to {filename}")
        return True
    except Exception as e:
        print(f"Error saving text to {filename}: {e}")
        return False

def save_chunks_for_inspection(chunks, filename):
    try:
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("CHUNKS INSPECTION FILE\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Target chunk size: {CHUNK_SIZE} characters\n")
            f.write(f"Chunk overlap: {CHUNK_OVERLAP} characters\n")
            f.write("="*80 + "\n\n")
            for chunk_index, chunk in enumerate(chunks, 1):
                chunk_chars = len(chunk)
                chunk_words = len(chunk.split())
                f.write(f"{'='*20} CHUNK {chunk_index} {'='*20}\n")
                f.write(f"Characters: {chunk_chars}\n")
                f.write(f"Words: {chunk_words}\n")
                f.write("-"*60 + "\n")
                f.write(chunk)
                f.write("\n")
                f.write("-"*60 + "\n")
                f.write(f"END OF CHUNK {chunk_index}\n\n")
        print(f"✅ Chunks saved successfully to {filename}.txt")
        return True
    except Exception as e:
        print(f"Error saving chunks to {filename}: {e}")
        return False

def save_resolved_chunks_for_inspection(resolved_chunks, filename):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("RESOLVED SELECTED CHUNKS\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total resolved: {len(resolved_chunks)}\n")
            f.write("="*80 + "\n\n")
            for chunk_index, resolved_chunk in resolved_chunks:
                chunk_chars = len(resolved_chunk)
                chunk_words = len(resolved_chunk.split())
                f.write(f"{'='*20} CHUNK {chunk_index} {'='*20}\n")
                f.write(f"Characters: {chunk_chars}\n")
                f.write(f"Words: {chunk_words}\n")
                f.write("-"*60 + "\n")
                f.write(resolved_chunk)
                f.write("\n")
                f.write("-"*60 + "\n")
                f.write(f"END OF CHUNK {chunk_index}\n\n")
        print(f"✅ Resolved chunks saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving resolved chunks: {e}")
        return False

def parse_resolved_chunks_from_file(filename):
    resolved_chunks = {}
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    chunk_splits = re.split(r"={10,} CHUNK (\d+) ={10,}", data)
    iter_chunks = iter(chunk_splits[1:])
    for chunk_num, rest in zip(iter_chunks, iter_chunks):
        chunk_index = int(chunk_num)
        m = re.search(r"-{60,}\n(.*?)\n-{60,}\nEND OF CHUNK", rest, re.DOTALL)
        chunk_text = m.group(1).strip() if m else ""
        resolved_chunks[chunk_index] = chunk_text
    return resolved_chunks

def parse_chunk_selection(user_input, total_chunks):
    if user_input.strip().lower() == "all":
        return list(range(1, total_chunks + 1))
    try:
        cleaned_input = user_input.strip()
        if cleaned_input.startswith('[') and cleaned_input.endswith(']'):
            chunk_numbers = ast.literal_eval(cleaned_input)
            if isinstance(chunk_numbers, list) and all(isinstance(x, int) for x in chunk_numbers):
                valid_chunks = [num for num in chunk_numbers if 1 <= num <= total_chunks]
                if len(valid_chunks) < len(chunk_numbers):
                    invalid_chunks = [num for num in chunk_numbers if num not in valid_chunks]
                    print(f"⚠️  Warning: Ignoring invalid chunk numbers: {invalid_chunks}")
                    print(f"   Valid chunk range is 1 to {total_chunks}")
                return sorted(valid_chunks) if valid_chunks else None
        else:
            chunk_numbers = [int(x.strip()) for x in cleaned_input.split(',')]
            valid_chunks = [num for num in chunk_numbers if 1 <= num <= total_chunks]
            if len(valid_chunks) < len(chunk_numbers):
                invalid_chunks = [num for num in chunk_numbers if num not in valid_chunks]
                print(f"⚠️  Warning: Ignoring invalid chunk numbers: {invalid_chunks}")
                print(f"   Valid chunk range is 1 to {total_chunks}")
            return sorted(valid_chunks) if valid_chunks else None
    except:
        return None

def parse_chunk_response(response_content):
    try:
        content = response_content.strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        elif content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
        parsed_data = json.loads(content)
        relationships_count = len(parsed_data.get('relationships', []))
        return parsed_data, relationships_count
    except json.JSONDecodeError as e:
        print(f"Error parsing chunk response: {e}")
        print(f"Response content: {response_content}")
        return None, 0
    except Exception as e:
        print(f"Unexpected error parsing chunk response: {e}")
        return None, 0

def user_chunk_selection(total_chunks):
    print("\n" + "="*60)
    print("CHUNK SELECTION")
    print("="*60)
    print(f"Total chunks available: {total_chunks}")
    print("\nPlease choose which chunks to process:")
    print("  - Type 'all' to process all chunks")
    print("  - Enter a list of chunk numbers like [1,3,5,7] or [2,4,6]")
    print("  - Or enter comma-separated numbers like 1,3,5,7")
    print(f"  - Valid chunk numbers are 1 to {total_chunks}")
    while True:
        chunk_selection_input = input("\nEnter your selection: ").strip()
        if chunk_selection_input.lower() == 'esc':
            print("Processing cancelled by user.")
            return None
        selected_chunks = parse_chunk_selection(chunk_selection_input, total_chunks)
        if selected_chunks is not None:
            print(f"\n✅ Selected {len(selected_chunks)} chunks to process: {selected_chunks}\n")
            return selected_chunks
        else:
            print("❌ Invalid input. Please enter 'all' or a list of chunk numbers.")
            print("   Examples: all, [1,2,3], [5,10,15], or 1,2,3")


def main():
    print("="*60)
    print("STARTING PROCESSING WITH CONTEXT-AWARE PRONOUN RESOLUTION AND NER")
    print(f"Using {CONTEXT_WINDOW} previous chunks for context")
    print("="*60)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    cleaned_filename = f"cleaned_text_{timestamp}.txt"
    chunks_filename = f"chunks_original_{timestamp}"

    # Build stage log lines with correct parameter sets
    stage1_log_line = get_model_log_line(
        1, STAGE1_MODEL, STAGE1_MODEL_PARAMS, STAGE1_REASONING_EFFORT, STAGE1_GPT5_PARAMS
    )
    stage2_log_line = get_model_log_line(
        2, STAGE2_MODEL, STAGE2_MODEL_PARAMS, STAGE2_REASONING_EFFORT, STAGE2_GPT5_PARAMS
    )

    # ----------- If resume from checkpoint --------------
    if os.path.exists(RESOLVED_TXT_FILENAME):
        print(f"\n'{RESOLVED_TXT_FILENAME}' found in the current directory.")
        print(f"Please review '{RESOLVED_TXT_FILENAME}' for accuracy.")
        print("Press Enter to continue with this file, or type 'new' to run from the beginning instead.")
        user_input = input().strip()
        if user_input.lower() == "new":
            print("\nUser requested new run from scratch. Starting from beginning...\n")
        else:
            resolved_chunks_dict = parse_resolved_chunks_from_file(RESOLVED_TXT_FILENAME)
            total_chunks_in_file = len(resolved_chunks_dict)

            print(f"\n##########")
            print(stage1_log_line)
            print("\n##########")
            print(stage2_log_line)

            print(f"Detected {total_chunks_in_file} chunks in the resolved text file.")

            selected_chunks = user_chunk_selection(total_chunks_in_file)
            if selected_chunks is None or len(selected_chunks) == 0:
                print("No chunks selected, exiting.")
                return

            chunks_with_texts = []
            for chunk_index in selected_chunks:
                resolved_text = resolved_chunks_dict.get(chunk_index)
                if (resolved_text is not None) and (resolved_text.strip() != ""):
                    chunks_with_texts.append((chunk_index, resolved_text))
                else:
                    print(f"⚠️  ERROR: No text found for chunk {chunk_index} in '{RESOLVED_TXT_FILENAME}'. Skipping this chunk.")

            if len(chunks_with_texts) == 0:
                print("❌ No valid resolved chunks to process for Stage 2. Exiting.")
                return

            print("\nStep 4: Performing NER on selected resolved chunks...\n")
            print(stage2_log_line)

            # Metadata
            final_structure = {
                "total_chunks": total_chunks_in_file,
                "chunks_processed": [],
                "chunks_selected": selected_chunks,
                "total_number_relationships": 0,
                "total_characters_processed": 0,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "context_window": CONTEXT_WINDOW,
                "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "model_stage1": get_model_metadata_dict(
                    STAGE1_MODEL, STAGE1_MODEL_PARAMS, STAGE1_REASONING_EFFORT, STAGE1_GPT5_PARAMS
                ),
                "model_stage2": get_model_metadata_dict(
                    STAGE2_MODEL, STAGE2_MODEL_PARAMS, STAGE2_REASONING_EFFORT, STAGE2_GPT5_PARAMS
                ),
                "pronoun_resolution_applied": True,
                "pronoun_resolution_method": "context_aware_chunks",
                "system_message_stage2": ner_system_message,
                "user_prompt_template_stage2": ner_user_prompt_template,
                "pronoun_resolution_system_message": pronoun_resolution_system_message,
                "pronoun_resolution_prompt_template": pronoun_resolution_prompt_template,
                "chunks": []
            }
            total_relationships = 0
            total_characters_processed = 0

            for chunk_index, resolved_chunk in chunks_with_texts:
                chunk_chars = len(resolved_chunk)
                total_characters_processed += chunk_chars
                print(f"Processing chunk {chunk_index} for NER...")
                print(f"  Chunk {chunk_index} characters: {chunk_chars}")
                print(f"  Making API call for NER on chunk {chunk_index}...")
                chunk_output = run_model_call(
                    model=STAGE2_MODEL,
                    system_message=ner_system_message,
                    user_prompt=ner_user_prompt_template.format(chunk=resolved_chunk),
                    model_params=STAGE2_MODEL_PARAMS,
                    reasoning_effort=STAGE2_REASONING_EFFORT,
                    gpt5_params=STAGE2_GPT5_PARAMS
                )
                print(f"  ✅ NER API call completed for chunk {chunk_index}")
                parsed_data, relationships_count = parse_chunk_response(chunk_output)
                if parsed_data:
                    print(f"  Chunk {chunk_index} processed successfully:")
                    print(f"    Relationships found: {relationships_count}")
                    print(f"    Characters processed: {chunk_chars}\n")
                    chunk_data = {
                        "chunk_number": chunk_index,
                        "total_relationships": relationships_count,
                        "chunk_characters": chunk_chars,
                        "resolved_chunk_text": resolved_chunk,
                        "relationships": parsed_data.get('relationships', [])
                    }
                    final_structure["chunks_processed"].append(chunk_index)
                    final_structure["chunks"].append(chunk_data)
                    total_relationships += relationships_count
                else:
                    print(f"  ❌ Could not parse response for chunk {chunk_index}, saving raw output.\n")
                    chunk_data = {
                        "chunk_number": chunk_index,
                        "total_relationships": 0,
                        "chunk_characters": chunk_chars,
                        "resolved_chunk_text": resolved_chunk,
                        "relationships": [],
                        "raw_response": chunk_output
                    }
                    final_structure["chunks_processed"].append(chunk_index)
                    final_structure["chunks"].append(chunk_data)

            final_structure["total_number_relationships"] = total_relationships
            final_structure["total_characters_processed"] = total_characters_processed

            output_filename = f"NER_results.json"
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(final_structure, f, indent=2, ensure_ascii=False)
            print("="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            print(stage1_log_line)
            print(stage2_log_line)
            print(f"Total chunks available: {total_chunks_in_file}")
            print(f"Chunks selected: {len(selected_chunks)}")
            print(f"Chunks processed: {len(final_structure['chunks_processed'])}")
            print(f"Total relationships found: {total_relationships}")
            print(f"Total characters processed: {total_characters_processed}")
            if len(final_structure['chunks_processed']) > 0:
                print(f"Average characters per chunk: {total_characters_processed/len(final_structure['chunks_processed']):.1f}\n")
            print(f"Output files:")
            print(f"  - Final results: {output_filename}")
            print(f"  - Resolved chunk text: {RESOLVED_TXT_FILENAME}")
            print("="*60)
            return

    #------------------- Full pipeline: no resolved_text.txt or user forced new ---------------------
    print("\nStep 1: Cleaning input text...")
    cleaned_text = clean_text(input_text)
    print(f"Original text length: {len(input_text)} characters")
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    save_text_for_inspection(cleaned_text, cleaned_filename)
    user_input = input(f"\nReview '{cleaned_filename}'. Press Enter to continue, or type 'esc' to quit: ")
    if user_input.lower() == 'esc':
        print("Processing cancelled by user.")
        return

    print("\nStep 2: Creating chunks from cleaned text...")
    chunks = chunk_input(cleaned_text, CHUNK_SIZE, CHUNK_OVERLAP)
    total_chunks = len(chunks)
    print(f"Created {total_chunks} chunks")
    save_chunks_for_inspection(chunks, chunks_filename)
    selected_chunks = user_chunk_selection(total_chunks)
    if selected_chunks is None or len(selected_chunks) == 0:
        print("No chunks selected, exiting.")
        return
    selected_chunk_indices = set(selected_chunks)

    print("Step 3: Resolving pronouns in selected chunks with rolling context window...\n")
    print(stage1_log_line)
    resolved_chunks = []
    for chunk_index, chunk in enumerate(chunks, 1):
        if chunk_index in selected_chunk_indices:
            print(f"Processing chunk {chunk_index}/{total_chunks}...")
            start_idx = max(0, chunk_index - 1 - CONTEXT_WINDOW)
            end_idx = chunk_index - 1
            previous_chunks = chunks[start_idx:end_idx]
            print(f"  Resolving pronouns in chunk {chunk_index} with {len(previous_chunks)} context chunks...")
            context_section = ""
            if previous_chunks:
                context_section = "CONTEXT (for reference only - do not modify):\n"
                for i, prev_chunk in enumerate(previous_chunks):
                    context_section += f"\n--- Previous Chunk {i+1} ---\n{prev_chunk}\n"
            else:
                context_section = "CONTEXT: No previous context available.\n"
            user_prompt = pronoun_resolution_prompt_template.format(
                context_section=context_section, main_text=chunk
            )
            resolved_chunk = run_model_call(
                model=STAGE1_MODEL,
                system_message=pronoun_resolution_system_message,
                user_prompt=user_prompt,
                model_params=STAGE1_MODEL_PARAMS,
                reasoning_effort=STAGE1_REASONING_EFFORT,
                gpt5_params=STAGE1_GPT5_PARAMS
            )
            print(f"  ✅ Pronoun resolution completed for chunk {chunk_index}")
            resolved_chunks.append((chunk_index, resolved_chunk))
        else:
            print(f"Skipping chunk {chunk_index} (not selected)")
    print(f"\nSaving {len(resolved_chunks)} resolved chunks...")
    save_resolved_chunks_for_inspection(resolved_chunks, RESOLVED_TXT_FILENAME)

    print("Review resolved chunks file. Press Enter to continue with NER, or type 'esc' to quit: ")
    user_input = input()
    if user_input.lower() == 'esc':
        print("Processing cancelled by user.")
        return

    print("\nStep 4: Performing NER on selected resolved chunks...\n")
    print(stage2_log_line)
    final_structure = {
        "total_chunks": total_chunks,
        "chunks_processed": [],
        "chunks_selected": selected_chunks,
        "total_number_relationships": 0,
        "total_characters_processed": 0,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "context_window": CONTEXT_WINDOW,
        "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "model_stage1": get_model_metadata_dict(
            STAGE1_MODEL, STAGE1_MODEL_PARAMS, STAGE1_REASONING_EFFORT, STAGE1_GPT5_PARAMS
        ),
        "model_stage2": get_model_metadata_dict(
            STAGE2_MODEL, STAGE2_MODEL_PARAMS, STAGE2_REASONING_EFFORT, STAGE2_GPT5_PARAMS
        ),
        "pronoun_resolution_applied": True,
        "pronoun_resolution_method": "context_aware_chunks",
        "original_text_length": len(input_text),
        "cleaned_text_length": len(cleaned_text),
        "input_text": input_text,
        "system_message_stage2": ner_system_message,
        "user_prompt_template_stage2": ner_user_prompt_template,
        "pronoun_resolution_system_message": pronoun_resolution_system_message,
        "pronoun_resolution_prompt_template": pronoun_resolution_prompt_template,
        "chunks": []
    }
    total_relationships = 0
    total_characters_processed = 0

    for chunk_index, resolved_chunk in resolved_chunks:
        original_chunk = chunks[chunk_index - 1]
        chunk_chars = len(resolved_chunk)
        total_characters_processed += chunk_chars
        print(f"Processing chunk {chunk_index} for NER...")
        print(f"  Chunk {chunk_index} characters: {chunk_chars}")
        print(f"  Making API call for NER on chunk {chunk_index}...")
        chunk_output = run_model_call(
            model=STAGE2_MODEL,
            system_message=ner_system_message,
            user_prompt=ner_user_prompt_template.format(chunk=resolved_chunk),
            model_params=STAGE2_MODEL_PARAMS,
            reasoning_effort=STAGE2_REASONING_EFFORT,
            gpt5_params=STAGE2_GPT5_PARAMS
        )
        print(f"  ✅ NER API call completed for chunk {chunk_index}")
        parsed_data, relationships_count = parse_chunk_response(chunk_output)
        if parsed_data:
            print(f"  Chunk {chunk_index} processed successfully:")
            print(f"    Relationships found: {relationships_count}")
            print(f"    Characters processed: {chunk_chars}")
            print(f"    Context chunks used: {min(chunk_index - 1, CONTEXT_WINDOW)}\n")
            chunk_data = {
                "chunk_number": chunk_index,
                "total_relationships": relationships_count,
                "chunk_characters": chunk_chars,
                "context_chunks_used": min(chunk_index - 1, CONTEXT_WINDOW),
                "original_chunk_text": original_chunk,
                "resolved_chunk_text": resolved_chunk,
                "relationships": parsed_data.get('relationships', [])
            }
            final_structure["chunks_processed"].append(chunk_index)
            final_structure["chunks"].append(chunk_data)
            total_relationships += relationships_count
        else:
            print(f"  ❌ Could not parse response for chunk {chunk_index}, saving raw output.")
            chunk_data = {
                "chunk_number": chunk_index,
                "total_relationships": 0,
                "chunk_characters": chunk_chars,
                "context_chunks_used": min(chunk_index - 1, CONTEXT_WINDOW),
                "original_chunk_text": original_chunk,
                "resolved_chunk_text": resolved_chunk,
                "relationships": [],
                "raw_response": chunk_output
            }
            final_structure["chunks_processed"].append(chunk_index)
            final_structure["chunks"].append(chunk_data)

    final_structure["total_number_relationships"] = total_relationships
    final_structure["total_characters_processed"] = total_characters_processed

    output_filename = f"NER_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_structure, f, indent=2, ensure_ascii=False)
    print("="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(stage1_log_line)
    print(stage2_log_line)
    print(f"Total chunks available: {total_chunks}")
    print(f"Chunks selected: {len(selected_chunks)}")
    print(f"Chunks processed: {len(final_structure['chunks_processed'])}")
    print(f"Total relationships found: {total_relationships}")
    print(f"Total characters processed: {total_characters_processed}")
    if len(final_structure['chunks_processed']) > 0:
        print(f"Average characters per chunk: {total_characters_processed/len(final_structure['chunks_processed']):.1f}\n")
    print(f"Output files:")
    print(f"  - Final results: {output_filename}")
    print("="*60)

if __name__ == "__main__":
    main()