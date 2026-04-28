import json

with open('20260410_parse_features_of_30_patients_wav2vec.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

functions_found = {}
for cell_idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_lines = cell['source']
        source_text = ''.join(source_lines)
        
        if 'def build_mfa_features' in source_text:
            functions_found['build_mfa_features'] = source_text
        if 'def load_mfa_alignments' in source_text:
            functions_found['load_mfa_alignments'] = source_text
        if 'def export_sentences_for_mfa' in source_text:
            functions_found['export_sentences_for_mfa'] = source_text

for func_name, code in functions_found.items():
    print(f"\n{'='*80}")
    print(f"FUNCTION: {func_name}")
    print('='*80)
    print(code)
