"""Normalise a spoken media title before asking RuTracker to search for it.

The tracker's `nm=` search is word-based: a declined form such as «Симпсонов»
returns only the site's search-help row, while «Симпсоны» returns the season
packs. This is a GENERAL rule over Russian declensions, not a list of titles:
pymorphy3 parses each token and sends its normal form, keeping the plural of a
plural inflected word («Симпсонов» -> «Симпсоны», not «симпсон»). An adjective
before a noun is inflected to agree with it, so «Черного зеркала» becomes
«Чёрное зеркало» rather than the unmatchable «Чёрный зеркало».

A token the analyser does not know is passed through unchanged. A missing
dependency is also a pass-through: normalisation must never break the download
path or turn an unfamiliar name into a different word.
"""
from __future__ import annotations

_MORPH = None
_MORPH_FAILED = False


def _morph():
    """Lazy, process-wide analyser — importing it costs dictionaries."""
    global _MORPH, _MORPH_FAILED
    if _MORPH is None and not _MORPH_FAILED:
        try:
            from pymorphy3 import MorphAnalyzer
            _MORPH = MorphAnalyzer()
        except Exception:
            _MORPH_FAILED = True
    return _MORPH


def _parse(morph, token):
    try:
        parsed = morph.parse(token)[0]
        return parsed if parsed.is_known else None
    except Exception:
        return None


# Finite set of download verbs; every conjugated form of these is brought to
# its infinitive so the narrow download route cannot be lost to tense/aspect.
_DOWNLOAD_VERBS = {"скачать", "скачивать", "загрузить", "загружать", "качать"}


def normalise_download_verb(text: str) -> str:
    """Bring the first download verb (if any) to infinitive, leave the rest."""
    morph = _morph()
    if morph is None:
        return text
    tokens = text.split()
    for i, token in enumerate(tokens[:3]):
        word = _parse(morph, token)
        # Past tense is a report («скачал серию»), never an order.
        if (word is not None and word.tag.POS == "VERB"
                and getattr(word.tag, "tense", "") != "past"
                and word.normal_form in _DOWNLOAD_VERBS):
            tokens[i] = word.normal_form
            break
    return " ".join(tokens)


def normal_query(query: str) -> str:
    """Return the outgoing search query with every known token lemmatised."""
    morph = _morph()
    if morph is None:
        return query
    tokens = query.split()
    parsed = [_parse(morph, token) for token in tokens]
    out = []
    for i, (token, word) in enumerate(zip(tokens, parsed)):
        if word is None:
            out.append(token)
            continue
        try:
            form = word.normal_form
            if word.tag.number == "plur":
                plural = word.inflect({"plur", "nomn"})
                if plural is not None:
                    form = plural.word
            # The title filter matches whole words: an adjective must agree
            # with the noun that follows it, not just be in nominative.
            if word.tag.POS == "ADJF" and i + 1 < len(parsed):
                noun = parsed[i + 1]
                if noun is not None and noun.tag.POS == "NOUN":
                    grammes = {"nomn", noun.tag.number}
                    if noun.tag.number == "sing" and noun.tag.gender:
                        grammes.add(noun.tag.gender)
                    lemma = _parse(morph, word.normal_form)
                    agreed = lemma.inflect(grammes) if lemma is not None else None
                    if agreed is not None:
                        form = agreed.word
            # A capitalised proper name must keep its STEM. pymorphy3 reads
            # «Влада» as a genitive of «Влад»; a strict-prefix lemma here is
            # a different name, not just a declined form of the same one.
            if (token[:1].isupper() and form.lower() != token.lower()
                    and token.lower().startswith(form.lower())):
                out.append(token)
                continue
            # Keep the writer's capitalisation: titles are proper names.
            if token[:1].isupper():
                form = form[:1].upper() + form[1:]
            out.append(form)
        except Exception:
            out.append(token)
    return " ".join(out)
