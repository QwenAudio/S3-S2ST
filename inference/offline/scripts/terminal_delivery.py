"""Deliver an existing finished callback even when its token count is unchanged.
Patch the terminal callback in the current process.
"""
import hashlib, inspect, textwrap, types, queue, contextlib
from pathlib import Path
EXPECTED="7620085de1dde462425775d69f8433965b6e4378ed152b8e9778432c40cea3f2"
OLD='''                    if output.finished:
                        logger.info("[%s] got same token num, maybe aborted",
                                    request_id)
                        break'''
NEW='''                    if output.finished:
                        # The consumer waits for finished, not a token-count change.
                        last_output = output
                        with suppress_output_queue_exception():
                            self.output_queue[request_id].put(output)
                        logger.info("[%s] delivered terminal same-token callback",
                                    request_id)
                        break'''
def install():
    from vllm.engine.omnidrca2_engine import OmniDRCA2Engine
    original=OmniDRCA2Engine._listen_thinker
    if getattr(original,"_terminal_delivery_fixed",False):return original
    path=Path(inspect.getsourcefile(original))
    import ast
    tree=ast.parse(path.read_text())
    node=next(f for c in tree.body if isinstance(c,ast.ClassDef) and c.name=="OmniDRCA2Engine" for f in c.body if isinstance(f,ast.FunctionDef) and f.name=="_listen_thinker")
    assert hashlib.sha256(ast.get_source_segment(path.read_text(),node).encode()).hexdigest()=="c8d10114407de0014e42ae0fbea0fafda5b6831d08b8f4d8d280b5d5efdb5729"
    source=inspect.getsource(original)
    assert source.count(OLD)==1
    namespace={}
    exec(compile(textwrap.dedent(source.replace(OLD,NEW)),str(Path(__file__)), "exec"),original.__globals__,namespace)
    patched=namespace["_listen_thinker"]
    patched._terminal_delivery_fixed=True
    OmniDRCA2Engine._listen_thinker=patched
    print("TERMINAL_DELIVERY_FIX_INSTALLED",EXPECTED,flush=True)
    return patched
def regression(patched):
    from types import SimpleNamespace as N
    class InlineThread:
        def __init__(self,target,daemon):self.target=target
        def start(self):self.target()
    class Logger:
        def __getattr__(self,name):return lambda *a,**k:None
    g=patched.__globals__.copy()
    g.update(SynchronizedGenerator=lambda generator,loop:iter(generator),Thread=InlineThread,logger=Logger(),suppress_output_queue_exception=contextlib.nullcontext)
    fn=types.FunctionType(patched.__code__,g)
    results=[]
    for lengths in [[(1,False),(1,True)],[(1,False),(1,False),(2,True)],[(0,True)]]:
        q=queue.Queue(); aligned=[]
        e=N(_loop=None,has_talker=True,talker_config=None,output_queue={"r":q},thinker_listener={},thinker_output_len={},time_profile=False,
            _is_cosyvoice_codex_talker_config=lambda c:True,_start_cosyvoice_aligned_talker=lambda r,o:aligned.append(o))
        outputs=[N(outputs=[N(token_ids=list(range(n)),text="x",finish_reason="stop" if done else None,stop_reason=None)],finished=done) for n,done in lengths]
        fn(e,"r",outputs,queue.Queue())
        delivered=list(q.queue)
        assert delivered[-1] is outputs[-1] and delivered[-1].finished
        assert len(aligned)==1 and aligned[0] is outputs[-1]
        assert sum(x.finished for x in delivered)==1
        results.append({"events":lengths,"terminal_delivered":True,"alignment_calls":len(aligned)})
    return results
