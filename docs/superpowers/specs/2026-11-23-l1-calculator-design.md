# L1 calculator — short design (approved 2026-11-23)

**Intent.** Answer arithmetic spoken in Russian entirely in L1, with no LLM and
no network. Recipes need percentages and addition; today «посчитай два процента
от того что на весах» had to be intercepted separately, and any other maths
went to L2, which has no calculator and wastes ~90 s.

**Approaches considered.**
1. Send maths to a chat model — rejected: slow (L2 90 s), and the model already
   failed on the kitchen percentage question.
2. Extend the scale-only percentage regex every time — rejected: it creates one
   narrow rule per phrasing (the «это» and «обнулились» history).
3. **Chosen:** a general, pure expression evaluator in `services/jev-router/calc.py`
   with a safe AST whitelist, intercepted before the classifier.

**Scope (user-approved).** Digits and Russian number words; `+ − × ÷`;
parentheses; decimals with a comma; percent (`15 процентов от 200`, `15% от 200`);
powers (`2^3`, `2 в степени 3`) and square root (`корень из 16`). Standard
precedence. No logarithms/trigonometry/dates yet.

**Guards.** Scale, time/date, weather, brightness/volume/battery percent, device
questions and definition questions («что такое корень») keep their current
routes. Calculator runs AFTER the scale interception and only when the text is
a parseable expression.

**Output.** One short Russian sentence with the numeric result; Edge/Coqui/Silero
num2ru is the single TTS choke point that turns digits into words.

**Files.** `services/jev-router/calc.py` (new), `resolver.py` (`resolve_calc`),
`app.py` (L0.6 interception + `route=calc` SSE), `tests/test_calc.py` (new).
