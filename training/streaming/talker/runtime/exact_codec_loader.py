"""Install before streaming training entry imports its loader wrappers.
No shared source modifications. Precomputed codec follows the online envelope path.
"""
import inspect,textwrap,hashlib

def patch_class(cls,namespace):
 if getattr(cls,"_exact_codec_loader_v1",False):return
 original=cls._prepare_single_data
 if original.__name__ != "_prepare_single_data":raise RuntimeError("Install exact codec loader before full-record wrappers")
 source=textwrap.dedent(inspect.getsource(original))
 if 'exact_codec_tokens' in source:return
 source=source.replace('@staticmethod\n','',1)
 needle='            codec_label_length = 0'
 # Dedent may change indentation; match the statement and preserve it.
 line=next(x for x in source.splitlines() if x.strip()=='codec_label_length = 0')
 source=source.replace(line+'\n',line+'\n'+line[:len(line)-len(line.lstrip())]+'exact_codec_tokens = []\n')
 old='''for x in slice_audio.split("|"):
                                    if x.startswith("c"):
                                        # discrete audio token id should be put in slice_ids_audio
                                        slice_ids_audio.append(int(x[1:]))'''
 # Work with relative indentation after dedenting the original class method.
 old=textwrap.dedent(old) if old in source else old
 start=source.index('for x in slice_audio.split("|"):')
 end=source.rfind('\n',start,source.index('# Keep offline codec',start))+1
 indent=len(source[:start].rsplit('\n',1)[-1]);base=' '*indent
 replacement='''exact_slice = [int(x[1:]) for x in slice_audio.split("|") if x.startswith("c")]
assert exact_slice and all(0 <= x < 59049 for x in exact_slice), "invalid precomputed codec"
assert slice.get("codec_contract") == "exact_slice_fsq125_v1", "unversioned offline codec rejected"
assert int(slice["speech_length"]) == len(exact_slice), "precomputed codec length mismatch"
exact_codec_tokens.extend(exact_slice)
slice_ids_audio.extend([-50] * len(exact_slice))
online_target_audio = True  # use alignment masks, not legacy per-WRITE text padding
codec_label_length += len(exact_slice)
if max_codec_label_length > 0 and codec_label_length > max_codec_label_length:
    raise ValueError("precomputed codec labels exceed pack limit")
'''
 source=source[:start]+replacement.replace('\n','\n'+base).rstrip()+'\n'+source[end:]
 marker=next(x for x in source.splitlines() if '# Megatron consumes next-token pairs' in x)
 pos=source.index(marker)
 restore_indent=marker[:len(marker)-len(marker.lstrip())]
 restore='''                    if exact_codec_tokens:
                        positions = [j for j, token in enumerate(input_ids_audio) if token == -50]
                        assert len(positions) == len(exact_codec_tokens), "mixed online/offline or envelope token mismatch"
                        for j, token in zip(positions, exact_codec_tokens):
                            assert labels_audio[j] == -50
                            input_ids_audio[j] = token
                            labels_audio[j] = token
'''
 restore=textwrap.indent(textwrap.dedent(restore),restore_indent)
 source=source[:pos]+restore+source[pos:]
 # Keep the real module globals: full-record streaming temporarily replaces
 # its audio loader here to serve cached waveform slices.
 ns=namespace;exec(compile(source,'<exact_codec_loader_v1>','exec'),ns)
 cls._prepare_single_data=staticmethod(ns['_prepare_single_data'])
 cls._exact_codec_loader_v1=True
 return source

def install():
 import funasr.datasets.minmo_datasets.datasets as dm
 return patch_class(dm.MinMo_S2S_Channels_SFT_Pack_Mega,vars(dm))
