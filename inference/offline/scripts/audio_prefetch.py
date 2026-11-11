"""Bounded, ordered CPU I/O prefetch. No model state or RNG is touched."""
from concurrent.futures import ThreadPoolExecutor

def ordered_prefetch(items,load,depth=4):
    if depth<=0:
        for item in items:yield item,load(item)
        return
    it=iter(items);pending=[]
    with ThreadPoolExecutor(max_workers=min(depth,4),thread_name_prefix='audio-prefetch') as pool:
        def submit():
            try:item=next(it)
            except StopIteration:return
            pending.append((item,pool.submit(load,item)))
        for _ in range(depth):submit()
        try:
            while pending:
                item,future=pending.pop(0)
                result=future.result();submit();yield item,result
        finally:
            for _,future in pending:future.cancel()
