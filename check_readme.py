import re

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

print("=== Markdown Structure Check ===")
code_blocks = re.findall(r'```', content)
print(f'Code blocks: {len(code_blocks)} (should be even)')
print(f'Balanced: {len(code_blocks) % 2 == 0}')

open_brackets = content.count('[')
close_brackets = content.count(']')
print(f'Brackets: [ = {open_brackets}, ] = {close_brackets}, balanced = {open_brackets == close_brackets}')

open_parens = content.count('(')
close_parens = content.count(')')
print(f'Parentheses: ( = {open_parens}, ) = {close_parens}, balanced = {open_parens == close_parens}')

print("\n=== Math Rendering Issues ===")
if '$$' in content:
    count = content.count('$$')
    print(f'WARNING: Found {count} instances of $$ (LaTeX blocks - not well supported by GitHub)')
    lines_with_double = [i+1 for i, line in enumerate(content.split('\n')) if '$$' in line]
    print(f'Lines with $$: {lines_with_double[:10]}')
else:
    print('No $$ found (good)')
    
dollar_count = content.count('$')
if dollar_count > 0:
    print(f'Found {dollar_count} dollar signs - checking for inline math...')
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '$' in line and '```' not in line:
            print(f'Line {i+1}: {line[:80]}')
            if i > 10:  # Only show first 10
                break
else:
    print('No $ found (good for GitHub rendering)')

print("\n=== Code Block Pairing ===")
lines = content.split('\n')
code_block_lines = [i+1 for i, line in enumerate(lines) if line.strip().startswith('```')]
if len(code_block_lines) % 2 == 0:
    print(f'Code blocks are balanced: {len(code_block_lines)//2} pairs')
else:
    print(f'ERROR: Unbalanced code blocks! Found at lines: {code_block_lines[:20]}')
