import re
from pathlib import Path

# Convert Linux paths to Windows paths
def convert_paths(content):
    return re.sub(r'/([a-zA-Z])/([^"\s\n\r]+)', r'\1:\\\2', content).replace('/', '\\')

# Process files
model_dir = Path("data/model/B04_2018_coldstart")
ext_files = ["DCSM-FM_3D_bc.ext", "DCSM-FM_3D_0_5nm.ext"]

for filename in ext_files:
    input_file = model_dir / filename
    output_file = model_dir / f"{input_file.stem}_windows{input_file.suffix}"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    converted_content = convert_paths(content)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(converted_content)
    
    print(f"Converted {filename} -> {output_file.name}")

print("Done. Update your MDU file to use the _windows.ext files.")