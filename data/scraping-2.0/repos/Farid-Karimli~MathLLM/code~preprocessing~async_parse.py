import os
from openai import AsyncOpenAI, BadRequestError, APIConnectionError, APIConnectionError, RateLimitError
import asyncio
from dotenv import load_dotenv
from tqdm import tqdm
import json
import re
import csv
import argparse
import tiktoken
# Set your OpenAI API key here

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY') 
client = AsyncOpenAI()

chunksize = 4096*2
prompt_tokens = 9102


function_schema = {
    "name": "parse_math_text",
    "description": "A function that takes as input a chapter from a math text and returns 4 dictionaries with theorems, definitions, corollaries, and propositions.",
    "parameters": {
        "type": "object",
        "properties": {
            "theorems": {
                "type": "object",
                "description": "Dictionary of theorems, keyed by theorem number and letter. Each value is an object with the theorem statement and its proof.",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The statement of the theorem."
                    },
                    "proof": {
                        "type": ["string", "null"],
                        "description": "The proof of the theorem. If the proof is not provided, this will be null."
                    }
                }
            },
            "definitions": {
                "type": "object",
                "description": "Dictionary of definitions, keyed by definition number and letter. Each value is the statement of the definition, including any notes on notation or common usage."
            },
            "corollaries": {
                "type": "object",
                "description": "Dictionary of corollaries, similar to theorems, keyed by corollary number and letter. Each value is an object with the corollary statement and its proof.",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The statement of the corollary."
                    },
                    "proof": {
                        "type": ["string", "null"],
                        "description": "The proof of the corollary. If the proof is not provided, this will be null."
                    }
                }
            },
            "propositions": {
                "type": "object",
                "description": "Dictionary of propositions, structured like theorems, keyed by proposition number and letter. Each value is an object with the proposition statement and its proof.",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The statement of the proposition."
                    },
                    "proof": {
                        "type": ["string", "null"],
                        "description": "The proof of the proposition. If the proof is not provided, this will be null."
                    }
                }
            }
        },
        "required": ["theorems", "definitions", "corollaries", "propositions"]
    }
}



with open('example_1.md', 'r') as f:
    content_example = f.read()

theorems_e = {"1.29": ["If $a$ and $b$ are real, then $(a, b)=a+b i$", "$$\n\\begin{aligned}\na+b i & =(a, 0)+(b, 0)(0,1) \\\\\n& =(a, 0)+(0, b)=(a, b) .\n\\end{aligned}\n$$"], 
            "1.31 (a)": ["If $z$ and $w$ are complex, then $\\overline{z+w}=\\bar{z}+\\bar{w}$", None],
            "1.31 (b)": ["If $z$ and $w$ are complex, then $\\overline{z w}=\\bar{z} \\cdot \\bar{w}$,$", None],
            "1.31 (c)": ["If $z$ and $w$ are complex, $z+\\bar{z}=2 \\operatorname{Re}(z), z-\\bar{z}=2 i \\operatorname{Im}(z)$,", None],
             "1.31 (d)": ["If $z$ and $w$ are complex, then $z \\bar{z}$ is real and positive (except when $z=0$ ).", "write $z=a+b i$, and note that $z \\bar{z}=a^{2}+b^{2}$."] }

definitions_e = {"1.30": "If $a, b$ are real and $z=a+b i$, then the complex number $\\bar{z}=a-b i$ is called the conjugate of $z$. The numbers $a$ and $b$ are the real part and the imaginary part of $z$, respectively.\n\nWe shall occasionally write\n\n$$\na=\\operatorname{Re}(z), \\quad b=\\operatorname{Im}(z)\n$$", 
               "1.32": "Definition If $z$ is a complex number, its absolute value $|z|$ is the nonnegative square root of $z \\bar{z}$; that is, $|z|=(z \\bar{z})^{1 / 2}$.\n\nThe existence (and uniqueness) of $|z|$ follows from Theorem 1.21 and part $(d)$ of Theorem 1.31.\n\nNote that when $x$ is real, then $\\bar{x}=x$, hence $|x|=\\sqrt{x^{2}}$. Thus $|x|=x$ if $x \\geq 0,|x|=-x$ if $x<0$."}

corollaries_e = {}

propositions_e = {}

example_output = {"theorems": theorems_e, "definitions": definitions_e, "corollaries": corollaries_e, "propositions": propositions_e}

example_output_empty = {"theorems": {}, "definitions": {}, "corollaries": {}, "propositions": {}}


with open('example_2.md', 'r') as f:
    example_4 = f.read()

output_4 = {'theorems': {}, 'definitions': {'2.18 (a)': 'For a metric space X and a point $p \\in X$:A neighborhood of $p$ is a set $N_{r}(p)$ consisting of all $q$ such that $d(p, q)<r$, for some $r>0$. The number $r$ is called the radius of $N_{r}(p)$.', '2.18 (b)': 'For a metric space X and a point $p \\in X$: A point $p$ is a limit point of the set $E$ if every neighborhood of $p$ contains a point $q \\neq p$ such that $q \\in E$.', '2.18 (c)': 'For a metric space X and a point $p \\in X$, and $E \\subset X$: If $p \\in E$ and $p$ is not a limit point of $E$, then $p$ is called an isolated point of $E$.', '2.18 (d)': 'For a metric space X and $E \\subset X$: $E$ is closed if every limit point of $E$ is a point of $E$.', '2.18 (e)': 'For a metric space X and a point $p \\in X$, and $E \\subset X$: A point $p$ is an interior point of $E$ if there is a neighborhood $N$ of $p$ such that $N \\subset E$.', '2.18 (f)': 'For a metric space X and $E \\subset X$: $E$ is open if every point of $E$ is an interior point of $E$.', '2.18 (g)': 'For a metric space X and points $p \\in X$, and $E \\subset X$: The complement of $E$ (denoted by $E^{c}$ ) is the set of all points $p \\in X$ such that $p \\notin E$.', '2.18 (h)': 'For a metric space X  $E \\subset X$: $E$ is perfect if $E$ is closed and if every point of $E$ is a limit point of $E$.', '2.18 (i)': 'For a metric space X and $E \\subset X$: $E$ is bounded if there is a real number $M$ and a point $q \\in X$ such that $d(p, q)<M$ for all $p \\in E$.', '2.18 (j)': 'For a metric space X and $E \\subset X$: $E$ is dense in $X$ if every point of $X$ is a limit point of $E$, or a point of $E$ (or both). Let us note that in $R^{1}$ neighborhoods are segments, whereas in $R^{2}$ neighborhoods are interiors of circles.'}, 'corollaries': {}, 'propositions': {}}

example_5 = '\n\n11.45 Theorem Let $\\left\\{\\phi_{n}\\right\\}$ be a complete orthonormal set. If $f \\in \\mathscr{L}^{2}(\\mu)$ and if\n\n$$\nf \\sim \\sum_{n=1}^{\\infty} c_{n} \\phi_{n}\n$$\n\nthen\n\n$$\n\\int_{X}|f|^{2} d \\mu=\\sum_{n=1}^{\\infty}\\left|c_{n}\\right|^{2}\n$$\n\nProof By the Bessel inequality, $\\Sigma\\left|c_{n}\\right|^{2}$ converges. Putting\n\n$$\ns_{n}=c_{1} \\phi_{1}+\\cdots+c_{n} \\phi_{n}\n$$\n\nthe Riesz-Fischer theorem shows that there is a function $g \\in \\mathscr{L}^{2}(\\mu)$ such that\n\n$$\ng \\sim \\sum_{n=1}^{\\infty} c_{n} \\phi_{n}\n$$\n\nand such that $\\left\\|g-s_{n}\\right\\| \\rightarrow 0$. Hence $\\left\\|s_{n}\\right\\| \\rightarrow\\|g\\|$. Since\n\n$$\n\\left\\|s_{n}\\right\\|^{2}=\\left|c_{1}\\right|^{2}+\\cdots+\\left|c_{n}\\right|^{2}\n$$\n\nwe have\n\n$$\n\\int_{X}|g|^{2} d \\mu=\\sum_{n=1}^{\\infty}\\left|c_{n}\\right|^{2}\n$$\n\nNow (106), (108), and the completeness of $\\left\\{\\phi_{n}\\right\\}$ show that $\\|f-g\\|=0$, so that (109) implies (107).\n\nCombining Theorems 11.43 and 11.45 , we arrive at the very interesting conclusion that every complete orthonormal set induces a 1-1 correspondence between the functions $f \\in \\mathscr{L}^{2}(\\mu)$ (identifying those which are equal almost everywhere) on the one hand and the sequences $\\left\\{c_{n}\\right\\}$ for which $\\Sigma\\left|c_{n}\\right|^{2}$ converges, on the other. The representation\n\n$$\nf \\sim \\sum_{n=1}^{\\infty} c_{n} \\phi_{n}\n$$\n\ntogether with the Parseval equation, shows that $\\mathscr{L}^{2}(\\mu)$ may be regarded as an infinite-dimensional euclidean space (the so-called "Hilbert space"), in which the point $f$ has coordinates $c_{n}$, and the functions $\\phi_{n}$ are the coordinate vectors.\n\n## EXERCISES\n\n1. If $f \\geq 0$ and $\\int_{E} f d \\mu=0$, prove that $f(x)=0$ almost everywhere on $E$. Hint: Let $E_{n}$ be the subset of $E$ on which $f(x)>1 / n$. Write $A=\\bigcup E_{n}$. Then $\\mu(A)=0$ if and only if $\\mu\\left(E_{n}\\right)=0$ for every $n$.\n2. If $\\int_{A} f d \\mu=0$ for every measurable subset $A$ of a measurable set $E$, then $f(x)=0$ almost everywhere on $E$.\n3. If $\\left\\{f_{n}\\right\\}$ is a sequence of measurable functions, prove that the set of points $x$ at which $\\left\\{f_{n}(x)\\right\\}$ converges is measurable.\n4. If $f \\in \\mathscr{L}(\\mu)$ on $E$ and $g$ is bounded and measurable on $E$, then $f g \\in \\mathscr{L}(\\mu)$ on $E$.\n5. Put\n\n$$\n\\begin{array}{rlrl}\ng(x) & = \\begin{cases}0 & \\left(0 \\leq x \\leq \\frac{1}{2}\\right), \\\\\n1 & \\left(\\frac{1}{2}<x \\leq 1\\right),\\end{cases} \\\\\nf_{2 k}(x) & =g(x) & (0 \\leq x \\leq 1), \\\\\nf_{2 k+1}(x) & =g(1-x) & (0 \\leq x \\leq 1)\n\\end{array}\n$$\n\nShow that\n\n$$\n\\liminf _{n \\rightarrow \\infty} f_{n}(x)=0 \\quad(0 \\leq x \\leq 1)\n$$\n\nbut\n\n$$\n\\int_{0}^{1} f_{n}(x) d x=\\frac{1}{2} .\n$$\n\n[Compare with (77).]\n\n6. Let\n\n$$\nf_{n}(x)= \\begin{cases}\\frac{1}{n} & (|x| \\leq n) \\\\ 0 & (|x|>n)\\end{cases}\n$$\n\nThen $f_{n}(x) \\rightarrow 0$ uniformly on $R^{1}$, but\n\n$$\n\\int_{-\\infty}^{\\infty} f_{n} d x=2 \\quad(n=1,2,3, \\ldots)\n$$\n\n(We write $\\int_{-\\infty}^{\\infty}$ in place of $\\int_{R 1}$.) Thus uniform convergence does not imply dominated convergence in the sense of Theorem 11.32. However, on sets of finite measure, uniformly convergent sequences of bounded functions do satisfy Theorem 11.32.\n\n7. Find a necessary and sufficient condition that $f \\in \\mathscr{R}(\\alpha)$ on $[a, b]$. Hint: Consider Example 11.6(b) and Theorem 11.33.\n8. If $f \\in \\mathscr{R}$ on $[a, b]$ and if $F(x)=\\int_{a}^{x} f(t) d t$, prove that $F^{\\prime}(x)=f(x)$ almost everywhere on $[a, b]$.\n9. Prove that the function $F$ given by (96) is continuous on $[a, b]$.\n10. If $\\mu(X)<+\\infty$ and $f \\in \\mathscr{L}^{2}(\\mu)$ on $X$, prove that $f \\in \\mathscr{L}(\\mu)$ on $X$. If\n\n$$\n\\mu(X)=+\\infty\n$$\n\nthis is false. For instance, if\n\n$$\nf(x)=\\frac{1}{1+|x|}\n$$\n\nthen $f \\in \\mathscr{L}^{2}$ on $R^{1}$, but $f \\notin \\mathscr{L}$ on $R^{1}$.\n\n11. If $f, g \\in \\mathscr{L}(\\mu)$ on $X$, define the distance between $f$ and $g$ by\n\n$$\n\\int_{x}|f-g| d \\mu\n$$\n\nProve that $\\mathscr{L}(\\mu)$ is a complete metric space.\n\n12. Suppose\n\n(a) $|f(x, y)| \\leq 1$ if $0 \\leq x \\leq 1,0 \\leq y \\leq 1$,\n\n(b) for fixed $x, f(x, y)$ is a continuous function of $y$,\n\n(c) for fixed $y, f(x, y)$ is a continuous function of $x$.\n\nPut\n\n$$\ng(x)=\\int_{0}^{1} f(x, y) d y \\quad(0 \\leq x \\leq 1)\n$$\n\nIs $g$ continuous?\n\n13. Consider the functions\n\n$$\nf_{n}(x)=\\sin n x \\quad(n=1,2,3, \\ldots,-\\pi \\leq x \\leq \\pi)\n$$\n\nas points of $\\mathscr{L}^{2}$. Prove that the set of these points is closed and bounded, but not compact.\n\n14. Prove that a complex function $f$ is measurable if and only if $f^{-1}(V)$ is measurable for every open set $V$ in the plane.\n15. Let $\\mathscr{R}$ be the ring of all elementary subsets of $(0,1]$. If $0<a \\leq b \\leq 1$, define\n\n$$\n\\phi([a, b])=\\phi([a, b))=\\phi((a, b])=\\phi((a, b))=b-a,\n$$\n\nbut define\n\n$$\n\\phi((0, b))=\\phi((0, b])=1+b\n$$\n\nif $0<b \\leq 1$. Show that this gives an additive set function $\\phi$ on $\\mathscr{R}$, which is not regular and which cannot be extended to a countably additive set function on a $\\sigma$-ring.\n\n16. Suppose $\\left\\{n_{k}\\right\\}$ is an increasing sequence of positive integers and $E$ is the set of all $x \\in(-\\pi, \\pi)$ at which $\\left\\{\\sin n_{k} x\\right\\}$ converges. Prove that $m(E)=0$. Hint: For every $A \\subset E$,\n\n$$\n\\int_{A} \\sin n_{k} x d x \\rightarrow 0\n$$\n\nand\n\n$$\n2 \\int_{A}\\left(\\sin n_{k} x\\right)^{2} d x=\\int_{A}\\left(1-\\cos 2 n_{k} x\\right) d x \\rightarrow m(A) \\quad \\text { as } k \\rightarrow \\infty \\text {. }\n$$\n\n17. Suppose $E \\subset(-\\pi, \\pi), m(E)>0, \\delta>0$. Use the Bessel inequality to prove that there are at most finitely many integers $n$ such that $\\sin n x \\geq \\delta$ for all $x \\in E$.\n18. Suppose $f \\in \\mathscr{L}^{2}(\\mu), g \\in \\mathscr{L}^{2}(\\mu)$. Prove that\n\n$$\n\\left|\\int f g d \\mu\\right|^{2}=\\int|f|^{2} d \\mu \\int|g|^{2} d \\mu\n$$\n\nif and only if there is a constant $c$ such that $g(x)=c f(x)$ almost everywhere. (Compare Theorem 11.35.)\n\n'

output_5 = {"theorems": {"11.45": ["Let $\\left\\{\\phi_{n}\\right\\}$ be a complete orthonormal set. If $f \\in \\mathscr{L}^{2}(\\mu)$ and if\n\n$$\nf \\sim \\sum_{n=1}^{\\infty} c_{n} \\phi_{n}\n$$\n\nthen\n\n$$\n\\int_{X}|f|^{2} d \\mu=\\sum_{n=1}^{\\infty}\\left|c_{n}\\right|^{2}\n$$\n\n", "By the Bessel inequality, $\\Sigma\\left|c_{n}\\right|^{2}$ converges. Putting\n\n$$\ns_{n}=c_{1} \\phi_{1}+\\cdots+c_{n} \\phi_{n}\n$$\n\nthe Riesz-Fischer theorem shows that there is a function $g \\in \\mathscr{L}^{2}(\\mu)$ such that\n\n$$\ng \\sim \\sum_{n=1}^{\\infty} c_{n} \\phi_{n}\n$$\n\nand such that $\\left\\|g-s_{n}\\right\\| \\rightarrow 0$. Hence $\\left\\|s_{n}\\right\\| \\rightarrow\\|g\\|$. Since\n\n$$\n\\left\\|s_{n}\\right\\|^{2}=\\left|c_{1}\\right|^{2}+\\cdots+\\left|c_{n}\\right|^{2}\n$$\n\nwe have\n\n$$\n\\int_{X}|g|^{2} d \\mu=\\sum_{n=1}^{\\infty}\\left|c_{n}\\right|^{2}\n$$\n\nNow (106), (108), and the completeness of $\\left\\{\\phi_{n}\\right\\}$ show that $\\|f-g\\|=0$, so that (109) implies (107)."]}, "definitions": {}, "corollaries": {}, "propositions": {}}

with open('example_3.md', 'r') as f:
    example_6 = f.read()

output_6 = {"theorems": {"4.11 (a)": ["For $M$, a closed subspace of a Hilbert space $H$: Every $x \\in H$ has then a unique decomposition\n\n$$\nx=P x+Q x\n$$\n\ninto a sum of $P x \\in M$ and $Q x \\in M^{\\perp}$", "suppose that $x^{\\prime}+y^{\\prime}=x^{\\prime \\prime}+y^{\\prime \\prime}$ for some vectors $x^{\\prime}, x^{\\prime \\prime}$ in $M$ and $y^{\\prime}, y^{\\prime \\prime}$ in $M^{\\perp}$. Then\n\n$$\nx^{\\prime}-x^{\\prime \\prime}=y^{\\prime \\prime}-y^{\\prime}\n$$\n\nSince $x^{\\prime}-x^{\\prime \\prime} \\in M, y^{\\prime \\prime}-y^{\\prime} \\in M^{\\perp}$, and $M \\cap M^{\\perp}=\\{0\\}$ [an immediate consequence of the fact that $(x, x)=0$ implies $x=0]$, we have $x^{\\prime \\prime}=x^{\\prime}, y^{\\prime \\prime}=y^{\\prime}$.\n\nTo prove the existence of the decomposition, note that the set\n\n$$\nx+M=\\{x+y: y \\in M\\}\n$$\n\nis closed and convex. Define $Q x$ to be the element of smallest norm in $x+M$; this exists, by Theorem 4.10. Define $P x=x-Q x$.\n\nSince $Q x \\in x+M$, it is clear that $P x \\in M$. Thus $P$ maps $H$ into $M$.\n\nTo prove that $Q$ maps $H$ into $M^{\\perp}$ we show that $(Q x, y)=0$ for all $y \\in M$. Assume $\\|y\\|=1$, without loss of generality, and put $z=Q x$. The minimizing property of $Q x$ shows that\n\n$$\n(z, z)=\\|z\\|^{2} \\leq\\|z-\\alpha y\\|^{2}=(z-\\alpha y, z-\\alpha y)\n$$\n\nfor every scalar $\\alpha$. This simplifies to\n\n$$\n0 \\leq-\\alpha(y, z)-\\bar{\\alpha}(z, y)+\\alpha \\bar{\\alpha} .\n$$\n\nWith $\\alpha=(z, y)$, this gives $0 \\leq-|(z, y)|^{2}$, so that $(z, y)=0$. Thus $Q x \\in M^{\\perp}$."], "4.11 (b)": ["For $M$, a closed subspace of a Hilbert space $H$, and the existing unique decomposition a unique decomposition \n\n$$\nx=P x+Q x\n$$\n\ninto a sum of $P x \\in M$ and $Q x \\in M^{\\perp}$: $P x$ and $Q x$ are the nearest points to $x$ in $M$ and in $M^{\\perp}$, respectively", "\n\nWe have already seen by $4.11 (a)$ that $P x \\in M$. If $y \\in M$, it follows that\n\n$$\n\\|x-y\\|^{2}=\\|Q x+(P x-y)\\|^{2}=\\|Q x\\|^{2}+\\|P x-y\\|^{2}\n$$\n\nwhich is obviously minimized when $y=P x$.\n\n"], "4.11 (c)": ["The mappings $P: H \\rightarrow M$ and $Q: H \\rightarrow M^{\\perp}$ are linear.\n\n", "If we apply $4.11 (a)$ to $x$, to $y$, and to $\\alpha x+\\beta y$, we obtain\n\n$$\nP(\\alpha x+\\beta y)-\\alpha P x-\\beta P y=\\alpha Q x+\\beta Q y-Q(\\alpha x+\\beta y) .\n$$\n\nThe left side is in $M$, the right side in $M^{\\perp}$. Hence both are 0 , so $P$ and $Q$ are linear."], "4.11 (d)": ["For $M$, a closed subspace of a Hilbert space $H$, and the existing unique decomposition a unique decomposition \n\n$$\nx=P x+Q x\n$$\n\ninto a sum of $P x \\in M$ and $Q x \\in M^{\\perp}$: $\\|x\\|^{2}=\\|P x\\|^{2}+\\|Q x\\|^{2}$", "\n\nSince $P x \\perp Q x,this follows from 4.11 (a)"]} , "definitions": {}, "corollaries": {"4.11": ["For $M$, a closed subspace of a Hilbert space $H$: If $M \\neq H$, then there exists $y \\in H, y \\neq 0$, such that $y \\perp M$.", "take $x \\in H, x \\notin M$, and put $y=Q x$. Since $P x \\in M, x \\neq P x$, hence $y=x-P x \\neq 0$"]}, "propositions": {} }

with open('example_4.md', 'r') as f:
    example_7 = f.read()

example_3 = '10.23 Theorem Suppose $T$ is a $\\mathscr{C}^{\\prime}$-mapping of an open set $E \\subset R^{n}$ into an open set $V \\subset R^{m}, S$ is a $\\mathscr{C}^{\\prime}$-mapping of $V$ into an open set $W \\subset R^{p}$, and $\\omega$ is a $k$-form in $W$, so that $\\omega_{S}$ is a $k$-form in $V$ and both $\\left(\\omega_{S}\\right)_{T}$ and $\\omega_{S T}$ are $k$-forms in $E$, where $S T$ is defined by $(S T)(\\mathbf{x})=S(T(\\mathbf{x}))$. Then\n\n$$\n\\left(\\omega_{S}\\right)_{T}=\\omega_{S T}\n$$\n\nProof If $\\omega$ and $\\lambda$ are forms in $W$, Theorem 10.22 shows that\n\n$$\n\\left((\\omega \\wedge \\lambda)_{S}\\right)_{T}=\\left(\\omega_{S} \\wedge \\lambda_{S}\\right)_{T}=\\left(\\omega_{S}\\right)_{T} \\wedge\\left(\\lambda_{S}\\right)_{T}\n$$\n\nand\n\n$$\n(\\omega \\wedge \\lambda)_{S T}=\\omega_{S T} \\wedge \\lambda_{S T}\n$$\n\nThus if (71) holds for $\\omega$ and for $\\lambda$, it follows that (71) also holds for $\\omega \\wedge \\lambda$. Since every form can be built up from 0 -forms and 1 -forms by addition and multiplication, and since (71) is trivial for 0 -forms, it is enough to prove (71) in the case $\\omega=d z_{q}, q=1, \\ldots, p$. (We denote the points of $E, V, W$ by $\\mathbf{x}, \\mathbf{y}, \\mathbf{z}$, respectively.)\n\nLet $t_{1}, \\ldots, t_{m}$ be the components of $T$, let $s_{1}, \\ldots, s_{p}$ be the components of $S$, and let $r_{1}, \\ldots, r_{p}$ be the components of $S T$. If $\\omega=d z_{q}$, then\n\n$$\n\\omega_{s}=d s_{q}=\\sum_{j}\\left(D_{j} s_{q}\\right)(\\mathbf{y}) d y_{j}\n$$\n\nso that the chain rule implies\n\n$$\n\\begin{aligned}\n\\left(\\omega_{S}\\right)_{T} & =\\sum_{j}\\left(D_{j} s_{q}\\right)(T(\\mathbf{x})) d t_{j} \\\\\n& =\\sum_{j}\\left(D_{j} s_{q}\\right)(T(\\mathbf{x})) \\sum_{i}\\left(D_{i} t_{j}\\right)(\\mathbf{x}) d x_{i} \\\\\n& =\\sum_{i}\\left(D_{i} r_{q}\\right)(\\mathbf{x}) d x_{i}=d r_{q}=\\omega_{S T} .\n\\end{aligned}\n$$'
output_3 = {"theorems": {"10.23": ['Suppose $T$ is a $\\mathscr{C}^{\\prime}$-mapping of an open set $E \\subset R^{n}$ into an open set $V \\subset R^{m}, S$ is a $\\mathscr{C}^{\\prime}$-mapping of $V$ into an open set $W \\subset R^{p}$, and $\\omega$ is a $k$-form in $W$, so that $\\omega_{S}$ is a $k$-form in $V$ and both $\\left(\\omega_{S}\\right)_{T}$ and $\\omega_{S T}$ are $k$-forms in $E$, where $S T$ is defined by $(S T)(\\mathbf{x})=S(T(\\mathbf{x}))$. Then\n\n$$\n\\left(\\omega_{S}\\right)_{T}=\\omega_{S T}\n$$\n\n', 'If $\\omega$ and $\\lambda$ are forms in $W$, Theorem 10.22 shows that\n\n$$\n\\left((\\omega \\wedge \\lambda)_{S}\\right)_{T}=\\left(\\omega_{S} \\wedge \\lambda_{S}\\right)_{T}=\\left(\\omega_{S}\\right)_{T} \\wedge\\left(\\lambda_{S}\\right)_{T}\n$$\n\nand\n\n$$\n(\\omega \\wedge \\lambda)_{S T}=\\omega_{S T} \\wedge \\lambda_{S T}\n$$\n\nThus if (71) holds for $\\omega$ and for $\\lambda$, it follows that (71) also holds for $\\omega \\wedge \\lambda$. Since every form can be built up from 0 -forms and 1 -forms by addition and multiplication, and since (71) is trivial for 0 -forms, it is enough to prove (71) in the case $\\omega=d z_{q}, q=1, \\ldots, p$. (We denote the points of $E, V, W$ by $\\mathbf{x}, \\mathbf{y}, \\mathbf{z}$, respectively.)\n\nLet $t_{1}, \\ldots, t_{m}$ be the components of $T$, let $s_{1}, \\ldots, s_{p}$ be the components of $S$, and let $r_{1}, \\ldots, r_{p}$ be the components of $S T$. If $\\omega=d z_{q}$, then\n\n$$\n\\omega_{s}=d s_{q}=\\sum_{j}\\left(D_{j} s_{q}\\right)(\\mathbf{y}) d y_{j}\n$$\n\nso that the chain rule implies\n\n$$\n\\begin{aligned}\n\\left(\\omega_{S}\\right)_{T} & =\\sum_{j}\\left(D_{j} s_{q}\\right)(T(\\mathbf{x})) d t_{j} \\\\\n& =\\sum_{j}\\left(D_{j} s_{q}\\right)(T(\\mathbf{x})) \\sum_{i}\\left(D_{i} t_{j}\\right)(\\mathbf{x}) d x_{i} \\\\\n& =\\sum_{i}\\left(D_{i} r_{q}\\right)(\\mathbf{x}) d x_{i}=d r_{q}=\\omega_{S T} .\n\\end{aligned}\n$$']}, "definitions": {}, "corollaries": {}, "propositions": {}}


def create_sample_resopnses(json_input):
    return {"role": "assistant", "content": None, "function_call": {"name": "parse_math_text", "arguments": json_input}}

incorrect_json_1 = '{"theorems": {"2.42": ["Every bounded infinite subset of $R^{k}$ has a limit point in $R^{k}$.", "Being bounded, the set $E$ in question is a subset of a $k$-cell $I \\subset R^{k}$. By Theorem 2.40, $I$ is compact, and so $E$ has a limit point in $I$, by Theorem 2.37."], "2.43": ["Let $P$ be a nonempty perfect set in $R^{k}$. Then $P$ is uncountable.", "Since $P$ has limit points, $P$ must be infinite. Suppose $P$ is countable, and denote the points of $P$ by $\\mathbf{x}_{1}, \\mathbf{x}_{2}, \\mathbf{x}_{3}, \\ldots$ We shall construct a sequence $\\left\\{V_{n}\\right\\}$ of neighborhoods, as follows.\\n\\nLet $V_{1}$ be any neighborhood of $\\mathbf{x}_{1}$. If $V_{1}$ consists of all $\\mathbf{y} \\in R^{k}$ such that $\\left|\\mathbf{y}-\\mathbf{x}_{1}\\right|<r$, the closure $\\bar{V}_{1}$ of $V_{1}$ is the set of all $\\mathbf{y} \\in R^{k}$ such that $\\left|\\mathbf{y}-\\mathbf{x}_{1}\\right| \\leq \\boldsymbol{r}$.\\n\\nSuppose $V_{"}'

corrected_json_1  = '{"theorems": {"2.42": ["Every bounded infinite subset of $R^{k}$ has a limit point in $R^{k}$.", "Being bounded, the set $E$ in question is a subset of a $k$-cell $I \\\\subset R^{k}$. By Theorem 2.40, $I$ is compact, and so $E$ has a limit point in $I$, by Theorem 2.37."], "2.43": ["Let $P$ be a nonempty perfect set in $R^{k}$. Then $P$ is uncountable.", "Since $P$ has limit points, $P$ must be infinite. Suppose $P$ is countable, and denote the points of $P$ by $\\\\mathbf{x}_{1}, \\\\mathbf{x}_{2}, \\\\mathbf{x}_{3}, \\\\ldots We shall construct a sequence $\\\\left\\\\{V_{n}\\\\right\\\\}$ of neighborhoods, as follows. Let $V_{1}$ be any neighborhood of $\\\\mathbf{x}_{1}$. If $V_{1}$ consists of all $\\\\mathbf{y} \\\\in R^{k}$ such that $\\\\left|\\\\mathbf{y}-\\\\mathbf{x}_{1}\\\\right|<r$, the closure $\\\\bar{V}_{1}$ of $V_{1}$ is the set of all $\\\\mathbf{y} \\\\in R^{k}$ such that $\\\\left|\\\\mathbf{y}-\\\\mathbf{x}_{1}\\\\right| \\\\leq \\\\boldsymbol{r}$. ..."]}, "definitions": {}, "corollaries": {}, "propositions": {}}'

incorrect_json_2 = '{"theorems": {}, "definitions": {"1.4": "Throughout Chap. 1, the set of all rational numbers will be denoted by $Q$." "1.5": "Let $S$ be a set. An order on $S$ is a relation, denoted by <, with the following two properties:\\n\\n(i) If $x \\in S$ and $y \\in S$ then one and only one of the statements is true.\\n\\n$$\\nx<y, \\quad x=y, \\quad y<x\\n$$\\n\\n(ii) If $x, y, z \\in S$, if $x<y$ and $y<z$, then $x<z$.\\n\\nThe statement \\" $x<y$ \\" may be read as \\" $x$ is less than $y$ \\" or \\" $x$ is smaller than $y$ \\" or \\" $x$ precedes $y$ \\". It is often convenient to write $y>x$ in place of $x<y$. The notation $x \\leq y$ indicates that $x<y$ or $x=y$, without specifying which of these two is to hold. In other words, $x \\leq y$ is the negation of $x>y$."}}'

corrected_json_2 ='{"theorems": {}, "definitions": {"1.4": "Throughout Chap. 1, the set of all rational numbers will be denoted by $Q$.", "1.5": "Let $S$ be a set. An order on $S$ is a relation, denoted by <, with the following two properties:\\n\\n(i) If $x \\\\in S$ and $y \\\\in S$ then one and only one of the statements\\n\\nis true.\\n\\n$$\\nx<y, \\\\quad x=y, \\\\quad y<x\\n$$\\n\\n(ii) If $x, y, z \\\\in S$, if $x<y$ and $y<z$, then $x<z$.\\n\\nThe statement \\"$x<y$\\" may be read as \\"$x$ is less than $y$\\" or \\"$x$ is smaller than $y$\\" or \\"$x$ precedes $y$\\".\\n\\nIt is often convenient to write $y>x$ in place of $x<y$.\\n\\nThe notation $x \\\\leq y$ indicates that $x<y$ or $x=y$, without specifying which of these two is to hold. In other words, $x \\\\leq y$ is the negation of $x>y$.", "1.6": "An ordered set is a set $S$ in which an order is defined.\\n\\nFor example, $Q$ is an ordered set if $r<s$ is defined to mean that $s-r$ is a positive rational number.", "1.7": "Suppose $S$ is an ordered set, and $E \\\\subset S$. If there exists a $\\\\beta \\\\in S$ such that $x \\\\leq \\\\beta$ for every $x \\\\in E$, we say that $E$ is bounded above, and call $\\\\beta$ an upper bound of $E$.\\n\\nLower bounds are defined in the same way (with $\\\\geq$ in place of $\\\\leq$ ).", "1.8": "Suppose $S$ is an ordered set, $E \\\\subset S$, and $E$ is bounded above. Suppose there exists an $\\\\alpha \\\\in S$ with the following properties:\\n\\n(i) $\\\\alpha$ is an upper bound of $E$.\\n\\n(ii) If $\\\\gamma<\\\\alpha$ then $\\\\gamma$ is not an upper bound of $E$.\\n\\nThen $\\\\alpha$ is called the least upper bound of $E."}, "corollaries": {}, "propositions": {}}'

async def fix_JSON(json_string, error):
    global incorrect_json_1, corrected_json_1, incorrect_json_2, corrected_json_2
    while True:
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo-1106", #"gpt-4-1106-preview"
                messages=[
                {"role": "system", "content": (f"You are a machine that takes as input incorrectly formatted JSON output "
                "with the error associated with them, and fixes and return the correct JSON output."
                "You must only return the corrected JSON output and nothing else! The JSON must also be formatted according to {function_schema}."
                "So if it missing any keys, you must fill them in and give them empty values.")},
                {"role": "user", "content": f"Return the correct JSON for this: {incorrect_json_1}. Here is the error: JSONDecodeError: Invalid \escape: line 1 column 167 (char 166)"},
                {"role": "assistant", "content": corrected_json_1},
                {"role": "user", "content": f"Return the correct JSON for this: {incorrect_json_2}. Here is the error: JSONDecodeError: Expecting ',' delimiter: line 1 column 119 (char 118)"},
                {"role": "assistant", "content": corrected_json_2},
                {"role": "user", "content": f" Return the correct JSON for this: {json_string}. Here is the error:{error}"}],
                temperature=0,
            )
            break
        except (APIConnectionError, APIConnectionError, RateLimitError) as e:
            await asyncio.sleep(60)

        except Exception as e:
            exit(1)
    return response.choices[0].message.content.strip(), response.choices[0].finish_reason

def ensure_max_token_input(chunk: str, ret: str):
    global chunksize, prompt_tokens
    """ensures the continution only uses maximum amount of tokens model can fit."""
    allowed_num_tokens = 128000 - prompt_tokens - 200 # - 200 because we add another prompt if continuing. just overestimate 
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding_both = encoding.encode(chunk + ret)
    start = max(0,len(encoding_both) - allowed_num_tokens)  #20,000 as an overestimate of prompt token usage. 
    start_chunk = encoding.decode(encoding.encode(chunk)[start:])
    return start_chunk, ret

async def extract_theorems(chapter_text, cont=None, model_type="gpt-3.5-turbo-1106"):
    global content_example, example_output, example_4, output_4, example_5, output_5, example_6, output_6, example_7
    messages_= [{"role": "system", "content": "You are a machine that takes as input chapters from a math text. \
                You must extract the relevant data from this input to use as arguments to pass into the given function provided.\
                If the theorem/corollary/definition/proposition has multiple parts, i.e. (a), (b), (c), etc., then \
                 you must parse the main statement and add the necessary information from it to each of the cases, and treat it as its own theorem/corollary/definition/proposition;\
                  e.g. Theorem 10 (a), Theorem 10 (b), etc. .\
                 Some corollaries may not be numbered, but they will be stated right after the theorem they follow from. In that case you must \
                 simply give the corollary the same number as the theorem it results from.\
                If a given theorem/corollary/definition/proposition is not necessarily a true statement on its own, but depends on the context of the text around it to make sense,\
                 You must add that context from the text so that the theorem/corollary/definition/proposition makes sense.\
                  If the chapter does not seem like it is from a math text, it may be the appendix, introduction\
                or some other part of the book that hasn't gotten to the material yet, then just pass in empty arguments\
                to the function and nothing else."},
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + content_example},
                create_sample_resopnses(json.dumps(example_output)),
                 {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_3},
                create_sample_resopnses(json.dumps(output_3)),
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_4},
                create_sample_resopnses(json.dumps(output_4)),
                {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_5},
                create_sample_resopnses(json.dumps(output_5)),
                 {"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + example_6},
                create_sample_resopnses(json.dumps(output_6))]
    if cont:
        messages_ += [{"role": "user", "content": "Parse, add, and extract the relevant data from this input to use as arguments to pass into the given function provided:" + chapter_text},
                create_sample_resopnses(json.dumps(cont)), {"role": "user", "content": "Please continue"}]
    else:
        messages_.append({"role": "user", "content": chapter_text})

    while True:
        try: 
            response =  await client.chat.completions.create(
                model=model_type,
                messages=messages_,
                functions=[function_schema],
                function_call={"name": "parse_math_text"},
                temperature=0,
            )
            break
        except (APIConnectionError, APIConnectionError, RateLimitError) as e: # handle ratelimit error
            await asyncio.sleep(60)
        except Exception as e:
            exit(1)
    if cont: 
        ret = cont + response.choices[0].message.function_call.arguments.strip()
    else:
        ret = response.choices[0].message.function_call.arguments.strip()

    return ret, response.choices[0].finish_reason


def string_to_dicts(ret_dict):
    # using get with empty dictionary in case gpt fails on output
    return ret_dict.get('theorems',{}), ret_dict.get('definitions', {}), ret_dict.get('corollaries', {}), ret_dict.get('propositions', {})


def safe_list_get(l, idx, default = "Error, no proof value given."):
  try:
    return l[idx]
  except Exception:
    return default

async def extract_correct_theorems(chunk, output_dir, reload = 0):
    #try getting JSON
    global example_output_empty
    count = 0
    retries = 0
    err_flag = 0

    if '(a)' in chunk or reload == 1 or get_chunks_2(chunk) > 0:
        model = "gpt-4-1106-preview" # gpt3.5-turbo cannot handle this case 
    else:
        model = "gpt-3.5-turbo-1106"
        
    ret, finish_reason = await extract_theorems(chunk, model_type=model)

    while count < 5:
        try:
            ret = json.loads(ret)
            break
        except Exception as e:
            error = e
            if finish_reason == 'length':
                try:
                    if retries == 0: 
                        retries += 1
                        ret, finish_reason = await extract_theorems(chunk, model_type="gpt-4-1106-preview")
                        continue
                    chunk, ret = ensure_max_token_input(chunk, ret)
                    ret,finish_reason  = await extract_theorems(chunk, cont = create_sample_resopnses(ret), model_type="gpt-4-1106-preview")
                except BadRequestError as e2: 
                    if e.code == 'context_length_exceeded':
                        with open(f"{output_dir}/context_length_exceeded.log", "a+") as logf:
                            logf.write(f'{e2=}, stop_reason = {finish_reason} for text: {chunk} \n' + u'\u2500' * 10)
                        return example_output_empty
            elif finish_reason == 'stop': # incorrect JSON output
                ret, finish_reason = await fix_JSON(ret, f"{e.__class__.__name__}: {e}")
                count += 1
    if reload == 0: 
        err_message_1 = "Error, no proof value given."
        err_message_2 = "No statement given"
        theorems_temp, definitions_temp, corollaries_temp, propositions_temp =string_to_dicts(ret)
        for (key1, value1),(key2, value2),(key3, value3), (key4, value4) in zip(theorems_temp.items(),definitions_temp.items(), corollaries_temp.items(), propositions_temp.items()) : 
            if safe_list_get(value1, 1) == err_message_1 or safe_list_get(value2, 1) == err_message_1\
                    or safe_list_get(value3, 1) == err_message_1 or safe_list_get(value4, 1) == err_message_1\
                        or safe_list_get(key1, 0) == err_message_2 or safe_list_get(key2, 0) == err_message_2\
                    or safe_list_get(key3, 0) == err_message_2 or safe_list_get(key4, 0) == err_message_2:
                ret = await extract_correct_theorems(chunk, output_dir, reload = 1)
                return ret
        reload = 1
    if reload == 1: 
        return string_to_dicts(ret)
    else:
        with open(f"{output_dir}/json_errors.log", "a+") as logf:
            logf.write(f'{error=}, stop_reason = {finish_reason} for text: {chunk} \n' + u'\u2500' * 10)
        return string_to_dicts(example_output_empty)

def get_chunks_2(document_content):
    # Pattern to match the start of each relevant section
    pattern = r"^\d+\.\d+ (Proposition|Corollary)"

    # Compile the regex pattern
    compiled_pattern = re.compile(pattern, re.MULTILINE)

    # Find all matches in the document content
    return len([m.start(0) for m in re.finditer(compiled_pattern, document_content)])


def get_chunks(document_content):
    # Pattern to match the start of each relevant section
    pattern = r"^\d+\.\d+ (Theorem|Definition|Definitions|Proposition|Corollary)"

    # Compile the regex pattern
    compiled_pattern = re.compile(pattern, re.MULTILINE)

    # Find all matches in the document content
    matches = [m.start(0) for m in re.finditer(compiled_pattern, document_content)]
    
    chunks = [document_content[start:end].strip() for start, end in zip(matches[:-1], matches[1:])]
    chunks.append(document_content[matches[-1]:])
    return chunks
    
  
async def process_md_files(step, folder_path, output_dir):
    global chunksize

    theorem_headers = ['Theorem', 'Statement', 'Proof']
    definition_headers = ['Definition', 'Statement']
    corollary_headers = ['Corollary', 'Statement', 'Proof']
    proposition_headers = ['Proposition', 'Statement', 'Proof']

    with open(f"{output_dir}/theorems.csv", 'a+', newline='', encoding='utf-8') as theorems_file, \
         open(f"{output_dir}/definitions.csv", 'a+', newline='', encoding='utf-8') as definitions_file, \
         open(f"{output_dir}/corollaries.csv", 'a+', newline='', encoding='utf-8') as corollaries_file, \
         open(f"{output_dir}/propositions.csv", 'a+', newline='', encoding='utf-8') as propositions_file:

        theorems_writer = csv.DictWriter(theorems_file, fieldnames=theorem_headers)
        definitions_writer = csv.DictWriter(definitions_file, fieldnames=definition_headers)
        corollaries_writer = csv.DictWriter(corollaries_file, fieldnames=corollary_headers)
        propositions_writer = csv.DictWriter(propositions_file, fieldnames=proposition_headers)

        theorems_writer.writeheader()
        definitions_writer.writeheader()
        corollaries_writer.writeheader()
        propositions_writer.writeheader()

        tasks = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.md'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Split content into chunks
                chunks = get_chunks(content)
                for chunk in chunks:
                    # Create and append the task
                    task = asyncio.create_task(extract_correct_theorems(chunk, output_dir, 0))
                    tasks.append(task)

        #current = current_process()
        pbar = tqdm(total=len(chunks), desc=f"Processing file {filename}", position=step + 1)

        # write to csv ask tasks complete  
        for task in asyncio.as_completed(tasks):
            theorems_temp, definitions_temp, corollaries_temp, propositions_temp = await task
            for key, value in theorems_temp.items(): 
                theorems_writer.writerow({'Theorem': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})
            for key, value in definitions_temp.items():
                definitions_writer.writerow({'Definition': key, 'Statement': value})
            for key, value in corollaries_temp.items():
                corollaries_writer.writerow({'Corollary': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})
            for key, value in propositions_temp.items():
                propositions_writer.writerow({'Proposition': key, 'Statement': safe_list_get(value, 0, "No statement given"), 'Proof': safe_list_get(value, 1)})

            pbar.update(1)

        pbar.close()

    return 


async def main(mathllm_folder, book, step):   
    await process_md_files(step, os.path.join(mathllm_folder, f'raw_data/{book}'), os.path.join(mathllm_folder, f'training_data/{book}'))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--book",  type=str)
    parser.add_argument("-p", "--path",  type=str)
    parser.add_argument("-s", "--step", type=int)
    args = parser.parse_args()

    asyncio.run(main(args.path,args.book, args.step))
    print('Done')
