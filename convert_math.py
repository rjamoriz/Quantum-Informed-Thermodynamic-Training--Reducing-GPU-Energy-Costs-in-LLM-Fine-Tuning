import re

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Convert $$ blocks to ```math blocks for better GitHub compatibility
def convert_math_blocks(text):
    # Pattern to match $$ ... $$ blocks
    pattern = r'\$\$\s*\n(.*?)\n\s*\$\$'
    
    def replacer(match):
        math_content = match.group(1)
        return f'```math\n{math_content}\n```'
    
    return re.sub(pattern, replacer, text, flags=re.DOTALL)

new_content = convert_math_blocks(content)

# Write back
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Converted all $$ blocks to ```math blocks for better GitHub rendering")
print(f"   Original length: {len(content)}")
print(f"   New length: {len(new_content)}")

# Count conversions
original_blocks = content.count('$$\n')
new_blocks = new_content.count('```math\n')
print(f"   Converted: {new_blocks} math blocks")
