import sys; import os; R_=os.environ.get('QROOT','competition'); sys.path[:0]=[R_+'/km',R_+'/verifiers']
from math import ceil
from dfthcsos.verify import heuristic_costs
N=76
def score(R,B,C,brot,lam_eff=49.3014,J=None):
    L=N+R*B; sel=(L-1).bit_length(); anc=sel-1
    Jmax=(128-sel-anc)//brot; J=J or min(Jmax,75); K=ceil(75/J)
    h=heuristic_costs(N,R,B,C,brot)
    T=2*K*(L-2)+h['toffoli']
    Q=h['persistent_qubits']+J*brot+max(h['temporary_qubits'],anc)
    return lam_eff*T*Q, T, Q, J, K
base=score(65,60,60,12)[0]
if __name__=='__main__':
    print('baseline', base)
    print('--- b_rot (same factorization)')
    for b in [8,9,10,11,12,13]:
        s,T,Q,J,K=score(65,60,60,b); print(f'brot {b}: J={J} K={K} T={T} Q={Q}  improvement x{base/s:.3f}')
    print('--- rank R*B (brot 12, C=60)')
    for R,B in [(65,60),(60,60),(55,60),(50,60),(45,60),(65,50),(65,40),(40,50)]:
        s,T,Q,J,K=score(R,B,60,12); print(f'R={R} B={B} L={76+R*B}: T={T} Q={Q} improvement x{base/s:.3f}')
    print('--- C (65,60,brot 12)')
    for C in [40,50,60,80]:
        s,T,Q,J,K=score(65,60,C,12); print(f'C={C}: T={T} Q={Q} improvement x{base/s:.3f}')
