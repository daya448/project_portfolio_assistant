# filter_requirements.py
pipreqs_file = "pipreqs.txt"
freeze_file = "freeze.txt"
output_file = "requirements.txt"

with open(pipreqs_file) as f:
    used = set(line.strip().split('==')[0].split('>=')[0].split('<=')[0].lower() for line in f if line.strip() and not line.startswith('#'))

with open(freeze_file) as f:
    lines = f.readlines()

with open(output_file, 'w') as f:
    for line in lines:
        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].lower()
        if pkg in used:
            f.write(line)
