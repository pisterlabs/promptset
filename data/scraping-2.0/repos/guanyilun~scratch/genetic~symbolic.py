#%%
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI
import os, re
from tqdm import tqdm
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

@dataclass(frozen=True)
class LLM:
    client: "LLM"
    model: str
    verbose: bool = False

    def get_response(self, prompt):
        completion = self.client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You can help me by answering my questions. You can also ask me questions.",
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model = self.model, 
        )
        response = completion.choices[0].message.content
        if self.verbose: print("Response:", response)
        return response


@dataclass
class LLMFuzzer:
    symbols: List[str]
    operators: List[str]
    llm: Optional["LLM"] = None
    
    def __post_init__(self):
        if self.llm is None:
            self.llm = LLM(client=OpenAI(), model="Zephyr")

    def parse_expr(self, expr_str):
        transformations = (standard_transformations + (implicit_multiplication_application,))
        expr = parse_expr(expr_str, transformations=transformations)
        return expr

    def rand_expr(self, n_expr: int = 20, npr: int = 20):
        prompt = "\n".join([
            f"Generate {npr} random mathematical expressions with a given list of symbols and operators. Give no explanations.",
            "",
            f"Symbols: {', '.join(self.symbols)}", 
            f"Operators: {', '.join(self.operators)}",
            "",
            "Expressions:"
        ])
        exprs = []
        pbar = tqdm(total=n_expr)
        while len(exprs) < n_expr:
            response = self.llm.get_response(prompt)

            # try parsing the response
            try:
                new_expr_strs = [re.sub(r"^\d+\.\s", "", line) for line in response.split("\n") if line]
            except:
                print("Failed to parse response")
                continue

            # try parsing the expressions
            new_exprs = []
            for expr_str in new_expr_strs:
                try:
                    new_expr = self.parse_expr(expr_str)
                    assert new_expr.free_symbols.issubset(set(sp.symbols(self.symbols)))
                    # new_expr = sp.expand(new_expr)
                except:
                    continue
                new_exprs.append(new_expr)
            pbar.update(len(new_exprs))
            exprs.extend(new_exprs)
        
        return exprs[:n_expr]

    def gen_expr_similar(self, *exprs, n_expr: int = 1, show_pbar: bool = True, max_retry: int = 3):
        """generate an expression (or more) that is similar to the given expressions"""
        prompt = "\n".join([
            "Given a set of mathematical expressions, generate a new expression similar to the given expressions. Give no explanation.",
            # "Given two mathematical expressions, give a best guess what an expression in between the two expressions should be. Give no explanation.",
            "",
            f"Symbols: {', '.join(self.symbols)}",
            "Expressions:"
        ] + [f"{expr}" for expr in exprs] + [
            "",
            "New expression:"
        ])
        response = self.llm.get_response(prompt)
        # get last non-empty line as expression in case an explanation is given
        response = [l for l in response.split("\n") if l][-1]

        exprs = []
        pbar = tqdm(total=n_expr, disable=not show_pbar)
        n_attempt = 0
        while len(exprs) < n_expr:
            try:
                expr = self.parse_expr(response)
                assert expr.free_symbols.issubset(set(sp.symbols(self.symbols)))
                # expr = sp.expand(expr)
            except:
                if n_attempt >= max_retry:
                    raise ValueError("Max retry exceeded")
                n_attempt += 1
                continue
            pbar.update(1)
            exprs.append(expr)
        if n_expr == 1:
            return exprs[0]
        else:
            return exprs[:n_expr]

#%%
if __name__ == "__main__":
    llm = LLM(client=OpenAI(), model="Zephyr")
    fuzzer = LLMFuzzer(
        symbols=["x", "y"], 
        operators=["+", "-", "*", "/"],
        llm=llm
    )
    exprs = fuzzer.rand_expr(n_expr=20)

    fuzzer.gen_expr_similar(*exprs[:2], show_pbar=False)
