WINDOW = (4, 6, True)
EARLY = 6
STREAM = 2
CHUNK_BITS = 26

VARIANTS = {
    'full': dict(plan='adaptive', early=EARLY, stream=STREAM, mix=None, late=None),
    'no-late-term-entry': dict(plan='adaptive', early=EARLY, stream=STREAM, mix=True, late=False),
    'restriction-only': dict(plan='adaptive', early=0, stream=0, mix=False, late=False),
    'full-width': dict(plan='full-width', early=0, stream=0, mix=False, late=False),
    'early-distribution-only': dict(plan='adaptive', early=EARLY, stream=0, mix=False, late=False),
    'overlap-only': dict(plan='adaptive', early=0, stream=STREAM, mix=True, late=False),
    'no-early-distribution': dict(plan='adaptive', early=0, stream=STREAM, mix=True, late=True),
}


def variant(name: str, gpus: int) -> dict:
    v = dict(VARIANTS[name])
    if v['late'] is None:
        v['late'] = gpus > 1
    v['fuse_obs'] = not v['late']
    return v


def window(spec: str):
    w = [int(x) for x in spec.split(',')] if spec else []
    if not w:
        return 0
    return w[0] if len(w) == 1 else (w[0], w[1], bool(w[2]))
