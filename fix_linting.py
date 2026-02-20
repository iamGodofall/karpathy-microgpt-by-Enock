#!/usr/bin/env python3
"""
Script to fix common linting issues across the microgpt codebase.
"""

import re
from pathlib import Path


def fix_ambiguous_variable_names(content):
    """Fix ambiguous variable name 'l' by replacing with 'logit' or 'item'."""
    # Replace 'for l in' patterns
    content = re.sub(r"for l in ", "for logit in ", content)
    # Replace '[l /' patterns
    content = re.sub(r"\[l / ", "[logit / ", content)
    content = re.sub(r"\[l /", "[logit /", content)
    # Replace 'lambda l:' patterns
    content = re.sub(r"lambda l:", "lambda item:", content)
    # Replace 'for i, l in' patterns
    content = re.sub(r"for i, l in ", "for i, logit in ", content)
    return content


def remove_unused_imports(content, unused_imports):
    """Remove specific unused imports."""
    for import_line in unused_imports:
        # Match 'from X import Y' or 'import Y'
        pattern = rf"^{re.escape(import_line)}$\n?"
        content = re.sub(pattern, "", content, flags=re.MULTILINE)
    return content


def fix_bare_excepts(content):
    """Fix bare except clauses."""
    # Replace 'except:' with 'except Exception:'
    content = re.sub(r"except\s*:", "except Exception:", content)
    return content


def fix_empty_fstrings(content):
    """Fix empty f-strings."""
    # Replace f"" with "" and f'' with ''
    content = re.sub(r'f(["\'])\1', r"\1\1", content)
    return content


def process_file(filepath, issues):
    """Process a single file and fix its issues."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Fix ambiguous variable names
    if any("E741" in issue for issue in issues):
        content = fix_ambiguous_variable_names(content)

    # Fix bare excepts
    if any("E722" in issue for issue in issues):
        content = fix_bare_excepts(content)

    # Fix empty f-strings
    if any("F541" in issue for issue in issues):
        content = fix_empty_fstrings(content)

    # Remove unused imports
    unused_imports = []
    for issue in issues:
        if "F401" in issue:
            # Extract import from issue message
            match = re.search(r"'([^']+)' imported but unused", issue)
            if match:
                import_name = match.group(1)
                # Try to find the import line
                for line in content.split("\n"):
                    if f"import {import_name}" in line or f"import {import_name} " in line:
                        unused_imports.append(line.strip())
                        break

    if unused_imports:
        content = remove_unused_imports(content, unused_imports)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    return False


def main():
    # Read the flake8 report
    issues_by_file = {}
    report_path = Path("flake8_report.txt")

    if not report_path.exists():
        print("No flake8_report.txt found. Run flake8 first.")
        return

    with open(report_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            parts = line.strip().split(":")
            if len(parts) >= 3:
                filepath = parts[0]
                # issue_code = parts[2].strip().split()[0]  # Not used currently
                if filepath not in issues_by_file:
                    issues_by_file[filepath] = []
                issues_by_file[filepath].append(line.strip())

    # Process each file
    fixed_count = 0
    for filepath, issues in issues_by_file.items():
        if Path(filepath).exists():
            if process_file(filepath, issues):
                fixed_count += 1
                print(f"Fixed: {filepath}")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
