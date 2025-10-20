# queries.py

import re
import regex
import itertools

# For every item in iterable, compute function(item) and return a flat list
flat_map = lambda function, iterable: list(itertools.chain(*map(function, iterable)))

# For a list of blocks, select only the 'basic' ones, those without parentheses
f_parse_basic_blocks = lambda block_list: [block.strip() for block in block_list if "(" not in block and ")" not in block]

# Parse expressions divided by '&' (AND), ',' (AND) or '|' (OR)
f_parse_binary_ops = lambda string: re.split('\\||&|,', string)

# Parse expressions inside parentheses, respecting the nested structure
f_parse_parentheses = lambda string: list(map(lambda x: x.group("inner"), regex.finditer(r'(?P<rec>\((?P<inner>(?:[^()]++|(?P>rec))*)\))', string)))

def parse_query(query):
    # Parse all 'basic' blocks: those not inside any parentheses
    basic_blocks = f_parse_basic_blocks(
        f_parse_binary_ops(query)
    )
    # Get blocks within parentheses
    parentheses_blocks = f_parse_parentheses(query)
    while len(parentheses_blocks):
        # Parse basic logic blocks (no parentheses) inside each parentheses-enclosed block
        basic_blocks += f_parse_basic_blocks(
            flat_map(f_parse_binary_ops, parentheses_blocks)
        )
        # Get the remaining parentheses blocks not yet parsed
        parentheses_blocks = flat_map(f_parse_parentheses, parentheses_blocks)

    # While loop ends when there is nothing else to be parsed within parentheses
    return basic_blocks

def format_query(query):
    # Just enclose each basic expression within parentheses
    basic_expressions = parse_query(query)
    for expr in basic_expressions:
        query = query.replace(expr, f"({expr})")
    # Replace multiple spaces by single, remove tab and new lines
    query = ' '.join(query.split()).replace("\n", "").replace("\t", "")
    return query.replace("\n", "").strip()
