"""Pure spoken-arithmetic evaluator for L1.

No LLM, no network, no `eval`: Russian number words and operator words become a
plain arithmetic expression, which is then parsed with `ast` and evaluated over
a whitelist of nodes. Anything that still contains letters is refused — that is
what keeps home questions («сколько процентов заряда») out of this route.

Shared with the worker when a recipe needs a calculation mid-step.
"""

import ast
import math
import re

_UNITS = {
    "ноль": 0, "ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2,
    "три": 3, "трих": 3, "трех": 3, "трёх": 3, "четыре": 4, "четырёх": 4,
    "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12, "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16, "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18, "девятнадцать": 19,
    "девятнадцати": 19,
}
_TENS = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
}
_HUNDREDS = {
    "сто": 100, "ста": 100, "двести": 200, "двухсот": 200,
    "триста": 300, "трехсот": 300, "трёхсот": 300, "четыреста": 400,
    "четырёхсот": 400, "четырехсот": 400, "пятьсот": 500, "шестьсот": 600,
    "семьсот": 700, "восемьсот": 800, "девятьсот": 900,
}
_SCALES = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "тысячу": 1000,
    "миллион": 1_000_000, "миллиона": 1_000_000, "миллионов": 1_000_000,
}
_ALL_WORDS = {**_UNITS, **_TENS, **_HUNDREDS, **_SCALES}
_WORD_RE = re.compile(
    r"\b(?:" + "|".join(sorted(_ALL_WORDS, key=len, reverse=True)) + r")"
    r"(?:\s+(?:" + "|".join(sorted(_ALL_WORDS, key=len, reverse=True)) + r"))*\b",
    re.IGNORECASE,
)

_PREFIX_RE = re.compile(
    r"^\s*(?:а\s+)?(?:посчитай|рассчитай|вычисли|подсчитай|скажи|сколько будет"
    r"|что получится|сколько составит|положь|добавь|возьми)\s+",
    re.IGNORECASE,
)
# Recipe units are labels on the number, not part of the arithmetic. Without
# stripping them the expression still contains Cyrillic and is refused — which
# is what made «200 г плюс 20 г» useless in the kitchen on 23.11.2026.
_UNIT_RE = re.compile(
    r"\b(?:грамм\w*|килограмм\w*|миллилитр\w*|литр\w*|ложек|ложки|ложка"
    r"|г|кг|мл|л)\b",
    re.IGNORECASE,
)
_OPS = (
    (re.compile(r"\bквадратный\s+корень\s+из\b", re.I), "sqrt"),
    (re.compile(r"\bкорень\s+из\b", re.I), "sqrt"),
    (re.compile(r"\bв\s+степени\b", re.I), "**"),
    (re.compile(r"\bумнож(?:ить|ь|ьте|им)\b(?:\s+на)?", re.I), "*"),
    (re.compile(r"\bраздели(?:ть|ь|те)?\b(?:\s+на)?", re.I), "/"),
    (re.compile(r"\bподели(?:ть|ь|те)?\b(?:\s+на)?", re.I), "/"),
    (re.compile(r"\bплюс\b|\bсложи(?:ть|те)?\b|\bприбавь(?:те)?\b", re.I), "+"),
    (re.compile(r"\bминус\b|\bвычти\b|\bвычесть\b|\bотними\b|\bотнять\b", re.I), "-"),
)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
)
_MAX_ABS = 1e12


def _words_to_number(phrase: str) -> str:
    total = 0.0
    current = 0.0
    for word in phrase.lower().split():
        value = _ALL_WORDS.get(word)
        if value is None:
            return phrase
        if value in _SCALES.values() and word in _SCALES:
            total += (current or 1.0) * value
            current = 0.0
        else:
            current += value
    result = total + current
    return str(int(result)) if result.is_integer() else str(result)


def _looks_like_math(text: str) -> bool:
    return bool(re.search(
        r"[+\-*/^()%]|\b(?:плюс|минус|умнож\w*|раздели\w*|подели\w*"
        r"|процент\w*|корень\w*|степен\w*|сложи\w*|вычти\w*"
        r"|посчитай|вычисли|сколько будет)\b",
        text or "", re.IGNORECASE,
    ))


def evaluate(text: str) -> float | None:
    """Return the numeric result, or None when the text is not an expression."""
    if not _looks_like_math(text):
        return None
    expr = (
        (text or "").lower()
        .replace("−", "-").replace("×", "*").replace("÷", "/").replace("^", "**")
    )
    while True:
        stripped = _PREFIX_RE.sub("", expr)
        if stripped == expr:
            break
        expr = stripped
    for pattern, repl in _OPS:
        expr = pattern.sub(repl, expr)
    # Roots: whatever follows "sqrt" up to the next operator becomes its operand.
    expr = re.sub(
        r"\bsqrt\s+([^\s+\-*/^()]+(?:\s+(?:"
        + "|".join(sorted(_ALL_WORDS, key=len, reverse=True))
        + r"))*)",
        lambda m: "(" + _words_to_number(m.group(1)) + ")**0.5",
        expr,
    )
    expr = _WORD_RE.sub(lambda m: _words_to_number(m.group(0)), expr)
    expr = _UNIT_RE.sub(" ", expr)
    # «2 и 3» / «200 г и 10%» are additions in a recipe, not prose. Only
    # between two numbers, so «включи свет и телевизор» never reaches here.
    expr = re.sub(r"(?<=\d)\s+и\s+(?=\d)", "+", expr)
    expr = re.sub(r"\bпроцент\w*\b", "%", expr)
    expr = re.sub(r"(\d+(?:[.,]\d+)?)\s*%\s*от\s*", r"(\1/100)*", expr)
    expr = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", r"(\1/100)", expr)
    expr = expr.replace(",", ".")
    if re.search(r"[а-яё]", expr):
        return None
    expr = re.sub(r"\s+", "", expr)
    if not expr or not re.fullmatch(r"[0-9.\-+*/() ]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if any(not isinstance(node, _ALLOWED_NODES) for node in ast.walk(tree)):
        return None

    def value(node):
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("not a number")
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        left, right = value(node.left), value(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 10:
                raise ValueError("power too large")
            return left ** right
        raise ValueError("unsupported operator")

    try:
        result = value(tree.body)
    except (ZeroDivisionError, ValueError, OverflowError, ArithmeticError):
        return None
    if not math.isfinite(result) or abs(result) > _MAX_ABS:
        return None
    return 0.0 if abs(result) < 1e-9 else result


def sentence(value: float) -> str:
    """One speakable line. num2ru in the TTS service reads the digits."""
    if float(value).is_integer():
        return f"Получается {int(value)}."
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return f"Получается {text.replace('.', ',')}."
