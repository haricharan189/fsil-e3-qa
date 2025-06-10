import ast
from collections import defaultdict

# Load and parse the Python file
with open("q_a_l2.py", "r") as f:
    tree = ast.parse(f.read())

doc_template_map = defaultdict(set)

# Traverse the AST for function definitions
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        func_name = node.name

        # Traverse the function body to find doc_num values in dictionaries
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Dict):
                keys = [k.s for k in subnode.keys if isinstance(k, ast.Str)]
                if "doc_num" in keys:
                    idx = keys.index("doc_num")
                    value_node = subnode.values[idx]
                    if isinstance(value_node, ast.Str):
                        doc_num = value_node.s
                        doc_template_map[doc_num].add(func_name)
                    elif isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                        doc_num = value_node.value
                        doc_template_map[doc_num].add(func_name)

# Find the doc_num with the most templates
most_templates_doc = max(doc_template_map.items(), key=lambda x: len(x[1]))

print("doc_num with the most templates:", most_templates_doc[0])
print("Number of templates:", len(most_templates_doc[1]))
print("Templates:", most_templates_doc[1])
