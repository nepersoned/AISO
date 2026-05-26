import json

file_path = 'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\yelpzip_sampling_abc_aiso.ipynb'
out_path = 'c:\\Users\\kevin\\OneDrive\\Desktop\\AISO\\yelpzip_sampling_abc_aiso.py'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(out_path, 'w', encoding='utf-8') as out:
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            out.write(source + '\n\n')
print("Extracted script to yelpzip_sampling_abc_aiso.py")
