"""Random gene subsets constrained by induced STATE graph size and degree distribution."""
import numpy as np
from numba import njit


@njit(cache=True)
def graph_degrees(ptr,indices,nodes,n):
    mask=np.zeros(n,np.bool_);mask[nodes]=True
    result=np.zeros(len(nodes),np.int32)
    for i,u in enumerate(nodes):
        for j in range(ptr[u],ptr[u+1]):
            if mask[indices[j]]:result[i]+=1
    return result


@njit(cache=True)
def discrepancies(hist,k,target_cdf,target_zero,edges,target_edges,dtol,ktol,ztol):
    cumulative=0.;ks=0.
    for j in range(len(hist)):
        cumulative+=hist[j]/k
        ks=max(ks,abs(cumulative-target_cdf[j]))
    density=abs(edges-target_edges)/max(target_edges,1.)
    isolation=abs(hist[0]/k-target_zero)
    score=density/dtol+ks/ktol+isolation/ztol
    return score,density,ks,isolation


@njit(cache=True)
def match_subset(ptr,indices,pool,initial,reference_degrees,reference_n,seed,max_steps,dtol,ktol,ztol):
    np.random.seed(seed)
    n=len(ptr)-1;k=len(initial)
    nodes=initial.copy();chosen=np.zeros(n,np.bool_);chosen[nodes]=True
    counts=np.zeros(n,np.int32)
    for u in nodes:
        for j in range(ptr[u],ptr[u+1]):counts[indices[j]]+=1
    hist_size=1
    for i in range(n):hist_size=max(hist_size,ptr[i+1]-ptr[i]+1)
    hist=np.zeros(hist_size,np.int32)
    for u in nodes:hist[counts[u]]+=1
    target_hist=np.zeros(hist_size,np.float64)
    for d in reference_degrees:target_hist[d]+=1/len(reference_degrees)
    target_cdf=np.cumsum(target_hist);zero=target_hist[0]
    target_edges=reference_degrees.sum()/2*k*max(k-1,0)/(reference_n*max(reference_n-1,1))
    edges=0
    for u in nodes:edges+=counts[u]
    edges=edges//2
    score,de,ks,ze=discrepancies(hist,k,target_cdf,zero,edges,target_edges,dtol,ktol,ztol)
    best=score;best_nodes=nodes.copy();best_metrics=np.array([de,ks,ze]);accepted=0
    changes=np.zeros(n,np.int8);touched=np.zeros(n,np.bool_);affected=np.empty(n,np.int32)
    for step in range(max_steps):
        if de<=dtol and ks<=ktol and ze<=ztol:return nodes,np.array([de,ks,ze]),step,accepted,True
        if len(pool)<=k:break
        pos=np.random.randint(k);u=nodes[pos];v=pool[np.random.randint(len(pool))]
        if chosen[v]:continue
        linked=0
        for j in range(ptr[u],ptr[u+1]):
            if indices[j]==v:linked=1;break
        new_edges=edges-counts[u]+counts[v]-linked
        new_density=abs(new_edges-target_edges)/max(target_edges,1.)
        if de>dtol and new_density>de and np.random.random()>.02:continue
        used=0
        for j in range(ptr[u],ptr[u+1]):
            w=indices[j]
            if chosen[w] and w!=u:
                if not touched[w]:touched[w]=True;affected[used]=w;used+=1
                changes[w]-=1
        for j in range(ptr[v],ptr[v+1]):
            w=indices[j]
            if chosen[w] and w!=u:
                if not touched[w]:touched[w]=True;affected[used]=w;used+=1
                changes[w]+=1
        proposed=hist.copy();proposed[counts[u]]-=1;proposed[counts[v]-linked]+=1
        for j in range(used):
            w=affected[j];proposed[counts[w]]-=1;proposed[counts[w]+changes[w]]+=1
        new_score,nd,nk,nz=discrepancies(proposed,k,target_cdf,zero,new_edges,target_edges,dtol,ktol,ztol)
        temperature=.08*(1-step/max_steps)+.002
        accept=new_score<score or np.random.random()<np.exp(min(0.,(score-new_score)/temperature))
        if accept:
            for j in range(ptr[u],ptr[u+1]):counts[indices[j]]-=1
            for j in range(ptr[v],ptr[v+1]):counts[indices[j]]+=1
            chosen[u]=False;chosen[v]=True;nodes[pos]=v;hist=proposed;edges=new_edges
            score=new_score;de=nd;ks=nk;ze=nz;accepted+=1
            if score<best:best=score;best_nodes=nodes.copy();best_metrics=np.array([de,ks,ze])
        for j in range(used):w=affected[j];changes[w]=0;touched[w]=False
    passed=best_metrics[0]<=dtol and best_metrics[1]<=ktol and best_metrics[2]<=ztol
    return best_nodes,best_metrics,max_steps,accepted,passed


def random_initial(pool,reference,global_degree,k,rng):
    boundaries=np.array([0,4,8,16,32,64,128,256,512,1024])
    groups=np.searchsorted(boundaries,global_degree,side='right')
    target=np.bincount(groups[reference],minlength=len(boundaries)+1)
    wanted=rng.multinomial(k,target/target.sum())
    selected=[]
    for group,number in enumerate(wanted):
        candidates=pool[groups[pool]==group]
        if number and len(candidates):selected.extend(rng.choice(candidates,min(number,len(candidates)),replace=False).tolist())
    if len(selected)<k:
        mask=np.ones(len(global_degree),bool);mask[selected]=False
        candidates=pool[mask[pool]]
        selected.extend(rng.choice(candidates,k-len(selected),replace=False).tolist())
    return np.asarray(selected,np.int32)
