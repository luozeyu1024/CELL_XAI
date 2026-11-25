"""Review source-only threshold evidence and freeze selected mass policies."""
import argparse,json
import pandas as pd
from cumulative_common import *
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--choices',required=True);p.add_argument('--datasets',nargs='+');a=p.parse_args();choices=json.loads(Path(a.choices).read_text(encoding='utf-8'));summary=pd.read_csv(FIG/'source_data/graph_summary.tsv',sep='\t');cost=pd.read_csv(FIG/'source_data/gpu_cost.tsv',sep='\t');path=FIG/'source_data/selected_policies.json';selected=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
 designs=jobs()
 if a.datasets and (unknown:=set(a.datasets)-{job['dataset'] for job in designs}):
  p.error('Unknown source datasets: '+', '.join(sorted(unknown)))
 for job in designs:
  ds=job['dataset'];name=key(job)
  if a.datasets and ds not in a.datasets:continue
  review=choices.get(name,choices.get(ds))
  if review is None:continue
  q=review['threshold'];assert q in Q and len(review['rationale'])>25
  stats=summary[summary.dataset.eq(ds)&summary.fold.eq(job['fold'])&summary.scope.eq('source_training_only')];gpu=cost[cost.dataset.eq(ds)&cost.fold.eq(job['fold'])]
  assert len(stats)==16 and len(gpu)==32
  chosen=gpu[gpu.threshold.eq(q)];assert chosen.status.eq('ok').all();assert chosen.peak_allocated_GiB.max()<chosen.total_device_GiB.min()*.75
  cells=read_cells(job);calibration=cells[cells.split.eq('train')].cell_id.tolist()
  row=dict(threshold=q,rationale=review['rationale'],selection_scope='original source-training cells only; no validation/test outcome tuning',n_calibration_cells=len(calibration),shared_across_graphs=RELATIONS,batch_size=32,observed_selected_peak_allocated_GiB=float(chosen.peak_allocated_GiB.max()),unattainable_policy='retain all positive eligible nodes and mark failure; no silent deletion',source_training_ids=str(Path(job['cohort_root'])/'data/cohorts'/ds/'cells.tsv'))
  if name in selected:
   for field in ['threshold','rationale','n_calibration_cells','shared_across_graphs','batch_size','source_training_ids']:assert selected[name][field]==row[field],'Policy already frozen; do not silently change trained trial'
   continue
  selected[name]=row
 path.write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print('FROZEN_POLICIES',len(selected),'of',len(designs))
if __name__=='__main__':main()


