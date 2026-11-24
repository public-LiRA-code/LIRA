import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lira.exchanges import _choose_early, compile_plan
from lira.planner import make_plan
from benchmarks.single_gpu import build, observable
from lira.scheme import variant

RPN = 4


def _split(rank_bits, near_bits):
    k = len(rank_bits)
    kn = sum(1 for j in rank_bits if j < near_bits)
    return 2 ** k - 2 ** kn, 2 ** kn - 1


def lightning(case, obs, n, G, rpn=RPN):
    g = G.bit_length() - 1
    m, nb = n - g, min(g, rpn.bit_length() - 1)
    _, gates = build(case, n, 0)
    plan = make_plan(n, gates, order=list(range(n)), full_width=True, keep_order=True)
    terms = observable(obs, n, plan, case)
    acc = {ph: dict(rounds_far=0, rounds_near=0, exec_far=0, exec_near=0, data_far=0.0, data_near=0.0)
           for ph in ('forward', 'backward')}

    def app(ph, bits, times=1):
        J = sorted({b - m for b in bits if b >= m})
        if not J:
            return
        far, near = _split(J, nb)
        a, k = acc[ph], len(J)
        a['rounds_far' if far else 'rounds_near'] += 2 * times
        a['exec_far'] += 2 * far * times
        a['exec_near'] += 2 * (near + 1) * times
        a['data_far'] += 2 * far * 2.0 ** -k * times
        a['data_near'] += 2 * near * 2.0 ** -k * times

    for gt in plan.gates:
        app('forward', gt.targets)
    for _, ps in terms:
        app('forward', list(ps))
    for _, ps in terms:
        for b in ps:
            app('backward', [b])
    train = [p >= 0 for p in plan.param_index]
    first = train.index(True)
    for i in range(len(plan.gates) - 1, first - 1, -1):
        gt = plan.gates[i]
        app('backward', gt.targets, 2)
        if train[i]:
            app('backward', gt.targets[-1:] if gt.name.lower() in ('crx', 'cry', 'crz', 'cp', 'cphase') else gt.targets)
    tot = {k: acc['forward'][k] + acc['backward'][k] for k in acc['forward']}
    return dict(acc, total=tot, m=m)


def lira(case, obs, n, G, rpn=RPN, name='full'):
    g = G.bit_length() - 1
    m, nb = n - g, min(g, rpn.bit_length() - 1)
    opt = variant(name, G)
    _, gates = build(case, n, 0)
    plan = make_plan(n, gates)
    early = _choose_early(plan, g, True, 1.0, True, opt['early'])
    steps, counts, _ = compile_plan(plan, g, True, 1.0, actions=False, joint=True, early=early,
                                    stream=opt['stream'])
    r = dict(rounds_far=0, rounds_near=0, sent=0.0, sent_far=0.0)

    def rnd(J, senders, frac_far, frac_all, wl):
        far = any(j >= nb for j in J)
        r['rounds_far' if far else 'rounds_near'] += 1
        r['sent'] += senders * frac_all * 2.0 ** wl
        r['sent_far'] += senders * frac_far * 2.0 ** wl

    for st in steps:
        if st[0] in ('jxchg', 'sx'):
            ranks, J, wl = st[1], st[2], st[3]
            far, near = _split(J, nb)
            rnd(J, len(ranks), far / 2 ** len(J), (far + near) / 2 ** len(J), wl)
        elif st[0] == 'xchg':
            pairs, oneway, wl = st[1], st[2], st[3]
            j = (pairs[0][1] - pairs[0][0]).bit_length() - 1
            rnd([j], len(pairs) * (1 if oneway else 2), 0.5 * (j >= nb), 0.5, wl)
        elif st[0] == 'spread':
            J, wl = st[1], st[2]
            far, near = _split(J, nb)
            rnd(J, 1, far / 2 ** len(J), (far + near) / 2 ** len(J), wl)
    unit = float(G) * 2 ** m
    return dict(variant=name, early=early, two_way=counts['exchanges'], one_way=counts['oneway'], spread_bits=counts.get('spread', 0),
                rounds_far=2 * r['rounds_far'], rounds_near=2 * r['rounds_near'],
                fwd_data=r['sent'] / unit, fwd_data_far=r['sent_far'] / unit, m=m)


def _row(results, name):
    f = os.path.join(results, name)
    if not os.path.exists(f):
        return None
    x = json.load(open(f))
    rows = x['rows'] if isinstance(x, dict) and 'rows' in x else x
    rows = [r for r in (rows if isinstance(rows, list) else [rows]) if isinstance(r, dict) and ('exchanges' in r or 'mpiprof_total' in r)]
    return rows[0] if rows else None


def check(results):
    ok, seen = True, 0
    print('lightning.gpu: counted vs measured swap executions per rank (far, near) and GiB sent to other nodes')
    for case in ('mqt_su2random', 'sel_4'):
        for G, n in ((32, 35), (64, 36), (128, 37)):
            row = _row(results, f'lgpu_{case}_meanz_g{G}_n{n}_bd.json')
            if row is None:
                continue
            seen += 1
            f0, t0 = row['mpiprof_forward'][0], row['mpiprof_total'][0]
            mf, mt = (int(f0['n_swap_far']), int(f0['n_swap_near'])), (int(t0['n_swap_far']), int(t0['n_swap_near']))
            gf, gt = round(f0['bytes_far'] / 2 ** 30), round(t0['bytes_far'] / 2 ** 30)
            L = lightning(case, 'meanz', n, G)
            gib = 2 ** L['m'] * 8 / 2 ** 30
            cf = (L['forward']['exec_far'], L['forward']['exec_near'])
            ct = (L['total']['exec_far'], L['total']['exec_near'])
            cg = (round(L['forward']['data_far'] * gib), round(L['total']['data_far'] * gib))
            same = cf == mf and ct == mt and cg == (gf, gt)
            ok &= same
            print(f'  {case:14s} {G:3d} GPUs {n}q: forward {cf} / {mf}, gradient {ct} / {mt}, GiB {cg} / {(gf, gt)}  {"equal" if same else "DIFFERENT"}')
    print('lira: counted vs the engine counters of the measured runs (two-way exchanges; data sent and sent to other nodes, local states)')
    for case, obs in (('sel_4', 'meanz'), ('tfim1dp_n/2', 'native'), ('vqc', 'meanz'), ('mqt_su2random', 'meanz')):
        for G, n in ((2, 31), (4, 32), (8, 33), (16, 34), (32, 35), (64, 36), (128, 37), (256, 38)):
            row = _row(results, f"lira_{case.replace('/', '-')}_{obs}_g{G}_n{n}.json")
            if row is None:
                continue
            seen += 1
            O = lira(case, obs, n, G)
            print(f'  {case:14s} {G:3d} GPUs {n}q: two-way {O["two_way"]} / {row["exchanges"]}, one-way {O["one_way"]} / {row["oneway"]}, '
                  f'forward data {O["fwd_data"]:.3f} (x3 = {3 * O["fwd_data"]:.3f}) / {row["moved_state_vectors"]:.3f}, '
                  f'far {O["fwd_data_far"]:.3f} (x3 = {3 * O["fwd_data_far"]:.3f}) / {row["moved_far"]:.3f}; rounds far {O["rounds_far"]} near {O["rounds_near"]}')
            ok &= O['two_way'] == row['exchanges']
    if not seen:
        print(f'  no measured runs in {results}')
    return ok and seen > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='compare the counts with the measured runs in --results')
    ap.add_argument('--results', default='.', help='directory of the result files (lira_* main runs, lgpu_*_bd runs)')
    ap.add_argument('--weak', action='store_true', help='30 qubits per GPU, 2-512 GPUs')
    ap.add_argument('--cases', default='sel_4:meanz,tfim1dp_n/2:native')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    if args.check:
        print('all equal' if check(args.results) else 'SOME DIFFER')
    if args.weak:
        rows = []
        for item in args.cases.split(','):
            case, obs = item.split(':')
            for g in range(1, 10):
                G, n = 2 ** g, 30 + g
                L, O = lightning(case, obs, n, G), lira(case, obs, n, G)
                rows.append(dict(case=case, obs=obs, n=n, gpus=G, lightning=L, lira=O))
                print(f'{case:12s} {G:3d} GPUs {n}q  lightning rounds far {L["total"]["rounds_far"]:5d} near {L["total"]["rounds_near"]:5d}, '
                      f'data far {L["total"]["data_far"]:7.1f} near {L["total"]["data_near"]:6.1f} | lira rounds far {O["rounds_far"]} near {O["rounds_near"]}, '
                      f'fwd data {O["fwd_data"]:.3f} far {O["fwd_data_far"]:.3f}', flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(rows, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
