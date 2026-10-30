import argparse
import json
import os

import clip
import torch


def load_concepts_from_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # Check for the new unified concept bank
    if 'concept_bank' in data:
        all_concepts = data['concept_bank']
    # Fallback just in case you run this on your old JSON file
    elif 'vessel_concepts' in data and 'aneurysm_concepts' in data:
        print("⚠️ Warning: Old JSON format detected. Flattening lists automatically.")
        all_concepts = data['vessel_concepts'] + data['aneurysm_concepts']
    else:
        raise KeyError("JSON must contain 'concept_bank' key.")

    return all_concepts

def generate_concept_embeddings(json_input, output_path, clip_model="ViT-L/14"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load the CLIP model
    print(f"Loading CLIP {clip_model}...")
    model, _ = clip.load(clip_model, device=device)

    # 2. Load and Tokenize
    concepts = load_concepts_from_json(json_input)
    print(f"Encoding {len(concepts)} concepts from {json_input}...")

    text_tokens = clip.tokenize(concepts).to(device)

    with torch.no_grad():
        # Encode and Normalize
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # 3. Save the matrix + the original strings for reference
    torch.save({
        'concept_names': concepts,
        'embeddings': text_features.cpu()
    }, output_path)

    print(f"✅ Success! Saved {text_features.shape} matrix to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/metadata/concepts.json")
    parser.add_argument("--output", type=str, default="data/metadata/concept_embeds.pt")
    parser.add_argument("--model", type=str, default="ViT-L/14",
                        help="CLIP model name, e.g. ViT-L/14 or ViT-B/32")
    args = parser.parse_args()

    if os.path.exists(args.input):
        # If a non-default model is used, embed the model name in the output path
        # e.g. concept_embeds.pt → concept_embeds_ViT-B-32.pt
        output_path = args.output
        if args.model != "ViT-L/14":
            base, ext = os.path.splitext(args.output)
            suffix = args.model.replace("/", "-")   # ViT-B/32 → ViT-B-32
            output_path = f"{base}_{suffix}{ext}"

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generate_concept_embeddings(args.input, output_path, clip_model=args.model)
    else:
        print(f"Error: Could not find {args.input}")