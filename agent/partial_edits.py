import re
import difflib

def apply_search_replace(content: str, search_block: str, replace_block: str) -> str:
    """
    Applies a search/replace block to the content.
    The search_block must match exactly (ignoring some whitespace variations if needed, 
    but for now let's be strict).
    """
    if not search_block.strip():
        # If search is empty, we might be appending or something, 
        # but usually it's better to have a clear search.
        return content

    # Use escaping for regex or just string.replace
    # String replace is safer for literal code
    if search_block in content:
        return content.replace(search_block, replace_block, 1)
    
    # Try with stripped versions if exact match fails
    s_stripped = search_block.strip()
    if s_stripped in content:
        # This is riskier because we might lose indentation context
        # Better to try to find the match with indentation
        lines = content.splitlines(keepends=True)
        s_lines = s_stripped.splitlines()
        
        for i in range(len(lines) - len(s_lines) + 1):
            match = True
            for j in range(len(s_lines)):
                if s_lines[j].strip() != lines[i+j].strip():
                    match = False
                    break
            if match:
                # Found a fuzzy match, replace it
                # We should try to preserve the indentation of the first line
                before = "".join(lines[:i])
                after = "".join(lines[i+len(s_lines):])
                return before + replace_block + after

    return content

def parse_partial_edit(text: str) -> list[tuple[str, str]]:
    """
    Parses multiple SEARCH/REPLACE blocks.
    Format:
    <<<<<<< SEARCH
    ...
    =======
    ...
    >>>>>>> REPLACE
    """
    pattern = re.compile(
        r"<<<<<<< SEARCH
(.*?)
=======
(.*?)
>>>>>>> REPLACE", 
        re.DOTALL
    )
    matches = pattern.findall(text)
    return matches

def apply_multi_edit(content: str, edits: list[tuple[str, str]]) -> str:
    new_content = content
    for search, replace in edits:
        new_content = apply_search_replace(new_content, search, replace)
    return new_content
