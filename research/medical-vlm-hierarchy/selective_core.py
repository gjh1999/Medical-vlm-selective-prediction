"""Dependency-free reference baselines and metrics. No learned/model outputs included."""
import math


class Taxonomy:
    def __init__(self, parent, root='root'):
        self.parent=dict(parent);self.root=root
        if root in parent:raise ValueError('Root must not have a parent')
        self.nodes=set(parent)|{root}
        if not set(parent.values())<=self.nodes:raise ValueError('Missing parent node')
        for start in self.nodes:
            seen=set();v=start
            while v!=root:
                if v in seen:raise ValueError('Cycle in taxonomy')
                seen.add(v);v=self.parent[v]
        self.leaves=sorted(self.nodes-set(parent.values()))
        if len(self.leaves)<2:raise ValueError('At least two leaves required')
        self.desc={v:set() for v in self.nodes}
        for leaf in self.leaves:
            seen=set();v=leaf
            while True:
                if v in seen:raise ValueError('Cycle in taxonomy')
                seen.add(v);self.desc[v].add(leaf)
                if v==root:break
                v=self.parent[v]

    def validate(self, probabilities):
        if set(probabilities)!=set(self.leaves):raise ValueError('Leaf probability keys mismatch')
        if any(not math.isfinite(p) or p<0 for p in probabilities.values()):raise ValueError('Invalid probability')
        if not math.isclose(sum(probabilities.values()),1,abs_tol=1e-6):raise ValueError('Not normalized')

    def flat(self, p, threshold=0):
        self.validate(p)
        leaf=max(self.leaves,key=lambda y:p[y])
        return leaf if p[leaf]>=threshold else self.root

    def climbing(self, p, threshold):
        self.validate(p)
        if not 0<=threshold<=1:raise ValueError('Invalid threshold')
        v=max(self.leaves,key=lambda y:p[y])
        while v!=self.root and sum(p[y] for y in self.desc[v])<threshold:
            v=self.parent[v]
        return v

    def two_stage(self, p, leaf_threshold, branch_threshold):
        """Choose the mode if confident, otherwise the highest-mass top-level branch if nontrivial."""
        self.validate(p)
        if not (0 <= leaf_threshold <= 1 and 0 <= branch_threshold <= 1):
            raise ValueError("Invalid threshold")
        leaf = max(self.leaves, key=p.get)
        if p[leaf] >= leaf_threshold:
            return leaf
        branches = sorted(n for n, parent in self.parent.items() if parent == self.root)
        masses = {n: math.fsum(p[y] for y in self.desc[n]) for n in branches}
        best = max(branches, key=masses.get)
        if best not in self.leaves and masses[best] >= branch_threshold:
            return best
        return self.root

    def lca(self, leaves):
        if not leaves:return self.root
        if not set(leaves)<=set(self.leaves):raise ValueError('Unknown leaf')
        v=sorted(leaves)[0]
        while not set(leaves)<=self.desc[v]:v=self.parent[v]
        return v

    def metrics(self, truth, predictions):
        if not truth or len(truth)!=len(predictions):raise ValueError('Empty/mismatched data')
        if not set(truth)<=set(self.leaves):raise ValueError('Unknown truth')
        if not set(predictions)<=self.nodes:raise ValueError('Unknown prediction')
        def calc(indices):
            node=[i for i in indices if predictions[i]!=self.root]
            leaf=[i for i in node if predictions[i] in self.leaves]
            errors=sum(truth[i] not in self.desc[predictions[i]] for i in node)
            leaf_errors=sum(truth[i]!=predictions[i] for i in leaf)
            utility=sum((1-math.log(len(self.desc[predictions[i]]))/math.log(len(self.leaves)))
                        for i in node if truth[i] in self.desc[predictions[i]])/len(indices)
            return {'n':len(indices),'node_n':len(node),'node_errors':errors,
                    'leaf_n':len(leaf),'leaf_errors':leaf_errors,
                    'node_coverage':len(node)/len(indices),'leaf_coverage':len(leaf)/len(indices),
                    'node_risk':errors/len(node) if node else None,
                    'leaf_risk':leaf_errors/len(leaf) if leaf else None,'correct_information_utility':utility}
        overall=calc(list(range(len(truth))))
        overall['per_class']={y:calc([i for i,t in enumerate(truth) if t==y]) for y in self.leaves if y in truth}
        return overall


def _binomial_cdf(k,n,p):
    if p==0:return 1.
    if p==1:return float(k>=n)
    logs=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)+i*math.log(p)+(n-i)*math.log1p(-p) for i in range(k+1)]
    m=max(logs)
    return min(1.,math.exp(m)*sum(math.exp(v-m) for v in logs))


def upper_bound(errors, accepted, delta):
    """One-sided exact binomial bound; caller must ensure independent units."""
    if not 0<delta<1 or not 0<=errors<=accepted:raise ValueError('Invalid counts/delta')
    if accepted==0:return None
    if errors==accepted:return 1.
    if errors==0:return -math.expm1(math.log(delta)/accepted)
    lo,hi=0.,1.
    for _ in range(70):
        mid=(lo+hi)/2
        if _binomial_cdf(errors,accepted,mid)>delta:lo=mid
        else:hi=mid
    return (lo+hi)/2


def conformal_quantile(scores, alpha):
    """Split-conformal finite-sample rank; infinity when n too small."""
    if not 0<alpha<1 or not scores:raise ValueError('Invalid calibration data')
    if any(not math.isfinite(x) for x in scores):raise ValueError('Nonfinite score')
    rank=math.ceil((len(scores)+1)*(1-alpha))
    return sorted(scores)[rank-1] if rank<=len(scores) else math.inf


def lac_set(probabilities, quantile):
    """LAC leaf set; coverage claim requires exchangeability, not arbitrary shift."""
    return [y for y,p in probabilities.items() if 1-p<=quantile]


def select_policy(taxonomy, truth, candidates, delta=.05, leaf_target=.1, node_target=.05):
    """Candidates fixed on dev; only select using calibration labels."""
    if not candidates or len(candidates)>16:raise ValueError('Need 1..16 fixed candidates')
    adjusted=delta/(2*len(candidates));audit=[]
    for name,pred in candidates.items():
        metrics=taxonomy.metrics(truth,pred)
        leaf=upper_bound(metrics['leaf_errors'],metrics['leaf_n'],adjusted)
        node=upper_bound(metrics['node_errors'],metrics['node_n'],adjusted)
        # No leaves is not evidence of a leaf-risk bound. Require both constraints.
        feasible=leaf is not None and node is not None and leaf<=leaf_target and node<=node_target
        audit.append({'name':name,'metrics':metrics,'leaf_upper':leaf,'node_upper':node,'feasible':feasible})
    feasible=[r for r in audit if r['feasible']]
    best=max(feasible,key=lambda r:(r['metrics']['correct_information_utility'],r['metrics']['leaf_coverage'],r['name'])) if feasible else None
    return {'selected':best['name'] if best else None,'status':'selected' if best else 'no_certifiable_nontrivial_policy',
            'per_check_delta':adjusted,'audit':audit,'guarantee_conditions':'fixed_candidates_and_iid_calibration_units_only'}
