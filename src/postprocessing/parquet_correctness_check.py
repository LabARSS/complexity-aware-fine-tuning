import pandas as pd
import os
import numpy as np



file_path = "data/out/distillation/temperature_changes/mmlu_synth_qwen3_b_t0_8.parquet" #  Example path

if file_path:
    print(f"Reading file from: {file_path}")
    df = pd.read_parquet(file_path)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Validate Top-Level Columns
    expected_cols = ['input', 'output']
    if not all(col in df.columns for col in expected_cols):
        print(f"WARNING: Expected columns {expected_cols} not found. Found: {df.columns.tolist()}")
    
    # 2. Extract and Validate Nested Structure
    try:
        sample_input = df['input'].iloc[0]
        sample_output = df['output'].iloc[0]
        
        # Helper to safely get keys
        def get_keys(obj):
            if isinstance(obj, dict): return list(obj.keys())
            if isinstance(obj, np.ndarray): return "numpy array" # Should be dict in parquet
            return type(obj)

        print("\n--- Schema Validation ---")
        print(f"Sample Input keys: {get_keys(sample_input)}")
        print(f"Sample Output keys: {get_keys(sample_output)}")
        
        # 3. Check for API Errors or Empty Responses
        print("\n--- Data Quality & Error Check ---")
        
        def check_row(row):
            inp = row['input']
            out = row['output']
            
            issues = []
            # Basic type checks
            if not isinstance(inp, (dict, np.ndarray)): issues.append(f"Input type {type(inp)}")
            if not isinstance(out, (dict, np.ndarray)): issues.append(f"Output type {type(out)}")
            
            # Check for error field in output
            if isinstance(out, dict):
                if 'error' in out and out['error']: issues.append(f"API Error: {out.get('error')}")
                if not out.get('thinking'): issues.append("Missing 'thinking'")
                if not out.get('raw_response'): issues.append("Missing 'raw_response'")
            
            return issues

        # Apply check to a sample if dataset is huge, or all if small
        df['issues'] = df.apply(check_row, axis=1)
        problematic_rows = df[df['issues'].apply(len) > 0]
        
        if len(problematic_rows) > 0:
            print(f"Found {len(problematic_rows)} problematic rows:")
            print(problematic_rows[['issues']].head())
        else:
            print("No structural issues or missing fields found.")

        # 4. Semantic Review (Print Samples)
        print("\n--- Semantic Sample (Random 3 rows) ---")
        for i, row in df.sample(3).iterrows():
            inp = row['input']
            out = row['output']
            
            q_text = inp.get('question') if isinstance(inp, dict) else inp
            thinking = out.get('thinking') if isinstance(out, dict) else out
            
            print(f"\n[Row {i}]")
            print(f"Question: {str(q_text)[:200]}...")
            print(f"Choices: {inp.get('choices') if isinstance(inp, dict) else 'N/A'}")
            print(f"Correct Answer: {inp.get('answer') if isinstance(inp, dict) else 'N/A'}")
            print("-" * 10)
            print(f"Thinking (First 300 chars):\n{str(thinking)[:300]}...")
            print("-" * 20)

    except Exception as e:
        print(f"Error analyzing structure: {e}")
        print("Raw content of first row:")
        print(df.iloc[0])

else:
    print("File not found. Check the path.")