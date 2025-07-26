from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

import tree_sitter_cpp
from tree_sitter import Language, Parser

TS_LANG_CPP = Language(tree_sitter_cpp.language())


def extract_data_from_file(filename):
    """Extract all data from JSON file."""
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def extract_key_value(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    text_only = [entry["text"] for entry in data]
    return text_only


def extract_label_value(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    label_only = [entry["label"] for entry in data]
    return label_only


class NormalizationAgent:
    _COMMENT_PAT = re.compile(r"/\*.*?\*/|//.*?$", re.DOTALL | re.MULTILINE)

    def __init__(
            self,
            *,
            keep_comments: bool = True,
            expand_macros: bool = False,
            clang_path: Optional[str] = None,
            style: Optional[str] = None,
    ) -> None:
        if expand_macros and not clang_path:
            raise ValueError("clang_path must be provided when expand_macros=True")

        self.keep_comments = keep_comments
        self.expand_macros = expand_macros
        self.clang_path = clang_path or shutil.which("clang")
        self.style = style

        # self._c_parser = Parser(TS_LANG_C)
        self._cpp_parser = Parser(TS_LANG_CPP)

    # ---------------------------- public API --------------------------------

    def run(
            self,
            src: str,
            *,
            file_type: str,
            vulnerable: Optional[int] = None,
    ) -> Dict[str, Any]:
        file_type = file_type.lower()
        if file_type not in {"c", "cpp", "c++", "cxx"}:
            raise ValueError("file_type must be 'c' or 'cpp'")

        if self.expand_macros:
            src = self._run_clang_preprocessor(src)

        src = self._normalise_whitespace(src)
        src = self._ensure_trailing_newline(src)

        if not self.keep_comments:
            src = self._strip_comments_preserve_lines(src)

        src = self._apply_clang_format(src)

        compact_ast, line_map = self.produce_ast_compact_json(
            src,
            file_type,
            keep_comments=self.keep_comments
        )

        result = {
            "clean_code": src,
            "compact_ast": {"ast": compact_ast, "line_map": line_map},
        }

        # Add vulnerable field if provided
        if vulnerable is not None:
            result["vulnerable"] = vulnerable

        return result

    # ----------------------- normalisation helpers -------------------------

    @staticmethod
    def _normalise_whitespace(code: str) -> str:
        # Convert tabs→4 spaces, unify line endings, strip trailing ws.
        lines = [l.expandtabs(4).rstrip() for l in code.splitlines()]
        return "\n".join(lines)

    @staticmethod
    def _ensure_trailing_newline(code: str) -> str:
        return code if code.endswith("\n") else code + "\n"

    def _strip_comments_preserve_lines(self, code: str) -> str:
        return re.sub(self._COMMENT_PAT, lambda m: " " * (m.end() - m.start()), code)

    # -------------------------- clang helpers ------------------------------

    def _run_clang_preprocessor(self, code: str) -> str:
        """Pipe the snippet through clang -E (macro expansion)."""
        if not self.clang_path:
            raise RuntimeError("clang not found; cannot preprocess")

        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            res = subprocess.run(
                [self.clang_path, "-E", "-P", tmp_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return res.stdout
        finally:
            os.unlink(tmp_path)

    def _apply_clang_format(self, code: str) -> str:
        if self.style is None:
            return code
        clang_format = shutil.which("clang-format")
        if clang_format is None:
            return code
        try:
            res = subprocess.run(
                [clang_format, f"-style={self.style}"],
                input=code,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            return res.stdout
        except subprocess.CalledProcessError:
            return code

    # --------------------------- AST helpers -------------------------------

    def _produce_ast_json(self, code: str, file_type: str) -> str:
        parser = self._cpp_parser if file_type in {"cpp", "c++", "cxx"} else self._c_parser
        tree = parser.parse(code.encode("utf8"))
        return json.dumps(self._node_to_dict(tree.root_node))

    def _node_to_dict(self, node) -> Dict[str, Any]:  # recursive
        return {
            "type": node.type,
            "start_point": node.start_point,
            "end_point": node.end_point,
            "children": [self._node_to_dict(c) for c in node.children],
        }

    def produce_ast_compact_json(
            self,
            code: str,
            file_type: str,
            *,
            keep_comments: bool = False,
            structural_only: bool = True,
    ) -> tuple[dict, dict[str, int]]:
        """
        Efficiently return a compact AST JSON plus a {tag: line_nr} map.
        """
        code_bytes = code.encode("utf8")
        parser = self._cpp_parser if file_type in ("cpp", "c++", "cxx") else self._c_parser
        ts_tree = parser.parse(code_bytes)
        root = ts_tree.root_node

        # 1) Locate the actual function_definition node and only traverse it
        func_node = None
        for c in root.named_children:
            if c.type == "function_definition":
                func_node = c
                break
        if func_node is None:
            func_node = root  # fallback to whole tree if no function found

        RELEVANT = {
            "function_definition", "if_statement", "for_statement", "while_statement",
            "switch_statement", "call_expression", "assignment_expression",
            "return_statement", "binary_expression", "unary_expression",
            "expression_statement", "declaration", "break_statement",
            "continue_statement", "goto_statement", "do_statement",
            "case_statement", "default_statement", "labeled_statement",
            "asm_statement", "throw_statement", "try_statement",
            "catch_clause", "for_range_loop", "else_clause",
        }
        IDLITS = ("identifier", "number_literal", "string_literal")

        id_ctr = 0
        line_map: dict[str, int] = {}

        def recurse(node) -> Optional[dict]:
            nonlocal id_ctr

            typ = node.type

            # skip comments if asked
            if not keep_comments and typ == "comment":
                return None

            # prune non‐structural wrappers
            if structural_only and typ not in RELEVANT and typ not in IDLITS:
                out: List[dict] = []
                for child in node.named_children:
                    cobj = recurse(child)
                    if cobj:
                        out.append(cobj)
                # flatten up one level
                return {"child": out} if out else None

            # build this node
            tag = f"{typ}#{id_ctr}";
            id_ctr += 1
            ln = node.start_point[0]
            line_map[tag] = ln

            obj: dict[str, Any] = {"tag": tag, "type": typ, "line": ln}

            # leaf: keep actual lexeme
            if typ in IDLITS:
                lex = code_bytes[node.start_byte:node.end_byte].decode("utf8", "ignore")
                obj["value"] = lex
                return obj

            # internal: recurse children exactly once
            kids: List[dict] = []
            for child in node.named_children:
                cobj = recurse(child)
                if cobj:
                    kids.append(cobj)
            if kids:
                obj["child"] = kids
            return obj

        compact_root = recurse(func_node)

        return compact_root or {}, line_map


# ----------------------------- CLI demo ------------------------------------
if __name__ == "__main__":
    json_file = 'cve_output.json'

    # Extract all data from the JSON file
    data_entries = extract_data_from_file(json_file)

    results = []
    agent = NormalizationAgent()

    # Process each entry with its vulnerable value
    for entry in data_entries:
        text = entry.get("text", "")
        vulnerable = entry.get("vulnerable", None)

        # Run the normalization agent with the vulnerable parameter
        result = agent.run(text, file_type="cpp", vulnerable=vulnerable)
        results.append(result)

    output_file = 'result_output.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Normalization result saved to {output_file}")