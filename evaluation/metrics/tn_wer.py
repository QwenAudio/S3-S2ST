#!/usr/bin/env python3
"""Single official symmetric-normalization, corpus WER/CER scorer.

Raw ASR is immutable and remains the input to ASR BLEU/COMET. Chinese uses
vendored WeText 0.1.8 FSTs (Apache-2.0); English uses Whisper's normalizer.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import re
import unicodedata
from pathlib import Path
import zhconv
import jiwer

TN_PROFILE = 'wetext_0.1.8_zh_tn_itn_whisper_en_corpus'

@lru_cache(maxsize=1)
def chinese_graphs():
    import kaldifst
    from vendor.wetext_min.token_parser import TokenParser
    root = Path(__file__).resolve().parent / 'vendor/wetext_min/fsts/zh'
    return {op: (kaldifst.TextNormalizer(str(root/op/'tagger.fst')),
                 kaldifst.TextNormalizer(str(root/op/'verbalizer.fst')),
                 TokenParser('zh', op)) for op in ('tn', 'itn')}

def transduce(text, op):
    if not text:
        return text
    tagger, verbalizer, parser = chinese_graphs()[op]
    tagged = tagger(text)
    if not tagged:
        raise ValueError(f'WeText {op} tag failure: {text!r}')
    result = verbalizer(parser.reorder(tagged)).strip()
    if not result:
        raise ValueError(f'WeText {op} verbalize failure: {text!r}')
    return result

@lru_cache(maxsize=1)
def english_normalizer():
    from whisper.normalizers import EnglishTextNormalizer
    return EnglishTextNormalizer()

@lru_cache(maxsize=100000)
def clean(text, lang):
    text = unicodedata.normalize('NFKC', text or '').strip()
    if lang not in ('zh', 'en'):
        raise ValueError(f'Unsupported language: {lang}')
    if lang == 'en':
        # Do not strip Whisper's numeric %, decimal, sign or currency symbols.
        return ' '.join(english_normalizer()(text).split())
    text = zhconv.convert(text, 'zh-cn').lower()
    # Join presentation spaces adjacent to Chinese, retaining ASCII word gaps.
    text = re.sub(r'(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])', '', text)
    # Spoken and written forms pass through the exact same TN -> ITN path.
    if re.search(r'\d', text):
        text = transduce(text, 'tn')
    text = transduce(text, 'itn')
    kept = []
    for i, ch in enumerate(text):
        if ch.isspace():
            continue
        before = text[i-1] if i else ''
        after = text[i+1] if i+1 < len(text) else ''
        numeric_mark = ((ch in './:' and before.isdigit() and after.isdigit())
                        or (ch in '+-' and after.isdigit()) or ch in '%‰')
        if unicodedata.category(ch).startswith('P') and not numeric_mark:
            continue
        kept.append(ch)
    return ' '.join(kept)

def edit_counts(ref, hyp):
    n = len(ref.split())
    if not n:
        return {'n':0, 'substitutions':0, 'deletions':0, 'insertions':len(hyp.split())}
    if hasattr(jiwer, 'process_words'):
        m = jiwer.process_words(ref, hyp)
        get = lambda k: getattr(m, k)
    else:
        m = jiwer.compute_measures(ref, hyp)
        get = lambda k: m[k]
    return dict(n=n, **{k:int(get(k)) for k in ('substitutions','deletions','insertions')})

def score_rows(rows, lang):
    details, audit = [], []
    totals = dict(n=0, substitutions=0, deletions=0, insertions=0)
    seen = set()
    for key, raw_ref, raw_hyp in rows:
        if key in seen:
            raise ValueError(f'Duplicate utterance: {key}')
        seen.add(key)
        ref, hyp = clean(raw_ref, lang), clean(raw_hyp, lang)
        c = edit_counts(ref, hyp)
        for k in totals: totals[k] += c[k]
        den = c['n'] or 1
        details.append((key, (c['substitutions']+c['deletions']+c['insertions'])/den,
                        ref, hyp, c['insertions']/den, c['deletions']/den, c['substitutions']/den))
        audit.append(dict(wav=key, reference_raw=raw_ref, asr_raw=raw_hyp,
                          reference_tn=ref, asr_tn=hyp, tn_profile=TN_PROFILE, **c))
    if not seen or not totals['n']:
        raise ValueError('Corpus has no reference tokens; WER/CER undefined')
    rate = 100*(totals['substitutions']+totals['deletions']+totals['insertions'])/totals['n']
    result = dict(metric='CER' if lang=='zh' else 'WER', score=rate, wer_cer=rate,
                  samples=len(seen), num_samples=len(seen), profile=TN_PROFILE,
                  error_rate_aggregation='corpus: sum(S+D+I)/sum(reference tokens)',
                  **totals, ins=100*totals['insertions']/totals['n'],
                  subs=100*totals['substitutions']/totals['n'],
                  **{'del':100*totals['deletions']/totals['n']})
    return details, audit, result

def read_rows(path):
    rows = []
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip(): continue
        cols = line.split('\t')
        if len(cols)<4: raise ValueError(f'{path}:{lineno}: malformed ASR row')
        rows.append((cols[0],cols[2],cols[3]))
    return rows

def rescore(source, output, report, audit_path, lang):
    source, output, report, audit_path = map(Path, (source,output,report,audit_path))
    if source.resolve() in {p.resolve() for p in (output,report,audit_path)}:
        raise ValueError('Never overwrite raw ASR')
    details, audit, result = score_rows(read_rows(source),lang)
    result['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    for p in (output,report,audit_path): p.parent.mkdir(parents=True,exist_ok=True)
    body=''.join('\t'.join(map(str,r))+'\n' for r in details)
    output.write_text(body)
    audit_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    report.write_text('wav_res\tres_wer\ttext_ref\ttext_res\tres_wer_ins\tres_wer_del\tres_wer_sub\n'+body+
                      f"{result['metric']}: {result['score']:.3f}, SUBS: {result['subs']:.3f}, DEL: {result['del']:.3f}, INS: {result['ins']:.3f}\n")
    (output.parent/'wer_corpus.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser()
    for name in ('input','output','report','audit','lang'): ap.add_argument('--'+name,required=True)
    a=ap.parse_args()
    print(json.dumps(rescore(a.input,a.output,a.report,a.audit,a.lang),ensure_ascii=False))

if __name__=='__main__': main()
