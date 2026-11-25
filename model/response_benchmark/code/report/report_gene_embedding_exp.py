"""Summarize paired gene-feature experiments at the established evaluation units."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from gene_embedding_benchmark_exp import ROOT,BENCH,OUT,VARIANTS,DEFAULT_VARIANTS,jobs,destination
from train_response_exp import metrics,write_json

FIG=ROOT/'work/transfer_analysis'
METRICS=['auroc','balanced_accuracy','auprc_resistant','auprc_sensitive','mcc','macro_f1','brier_score']


def scalar_metrics(frame):
    return {k:v for k,v in metrics(frame).items() if not isinstance(v,(dict,list))}


def grouped_rows(variant,allow_partial=False):
    records=[]
    for task in sorted({job['dataset'] for job in jobs() if job['family']=='grouped'}):
        folds=[job for job in jobs() if job['dataset']==task]
        expected=[destination(job,variant)/'within'/task/'gat_gine/predictions.tsv' for job in folds]
        ready=all(path.exists() and path.with_name('metrics.json').exists()
                  and 'checkpoint_reload_max_margin_error' in json.loads(path.with_name('metrics.json').read_text())
                  for path in expected)
        if not ready:
            if allow_partial: continue
            raise ValueError('Incomplete grouped predictions: '+task+'/'+variant)
        frames=[];fold_scores=[]
        for job in folds:
            path=destination(job,variant)/'within'/task/'gat_gine/predictions.tsv'
            frame=pd.read_csv(path,sep='\t',float_precision='round_trip')
            frame=frame[frame.split.eq('test')].copy();frame['outer_fold']=job['fold']
            frames.append(frame);fold_scores.append(scalar_metrics(frame))
        test=pd.concat(frames,ignore_index=True)
        if not test.cell_id.is_unique: raise ValueError('Repeated outer-test cell')
        design=json.loads((Path(folds[0]['cohort_root'])/'data/cohorts'/task/'data_sources.json').read_text())
        group=design['group_key']
        if test.groupby(group).outer_fold.nunique().gt(1).any(): raise ValueError('Outer group leakage')
        out=OUT/variant/'grouped'/task/'summary';out.mkdir(parents=True,exist_ok=True)
        test.to_csv(out/'oof_cell_predictions.tsv',sep='\t',index=False,float_format='%.17g')
        if group=='patient_id':
            if test.groupby(group).y.nunique().gt(1).any(): raise ValueError('Mixed patient labels')
            patients=test.groupby(group).agg(y=('y','first'),probability_resistant=('probability_resistant','mean'),n_cells=('cell_id','size')).reset_index()
            patients['margin_resistant_minus_sensitive']=patients.probability_resistant
            patients.to_csv(out/'oof_patient_predictions.tsv',sep='\t',index=False,float_format='%.17g')
            score=scalar_metrics(patients)
            score['n_evaluation_units']=score.pop('n_cells');score['unit']='patient'
            score['n_sensitive_patients']=score.pop('n_sensitive')
            score['n_resistant_patients']=score.pop('n_resistant')
            score['ranking_score']='mean_cell_probability_resistant';scope='patient_grouped_OOF_patients'
        else:
            score={key:float(np.mean([f[key] for f in fold_scores])) for key in METRICS}
            score.update(n_cells=len(test),n_evaluation_units=len(folds),unit='sample_set' if group=='sample_id' else 'biological_replicate')
            score.update(resistant_prevalence=float(np.mean([f['resistant_prevalence'] for f in fold_scores])),
                         n_sensitive=int(test.y.eq(0).sum()),n_resistant=int(test.y.eq(1).sum()),threshold=.5,
                         ranking_score='resistant_minus_sensitive_logit; equal outer-fold metric mean')
            scope='sample_set_macro' if group=='sample_id' else 'replicate_macro'
        records.append(dict(dataset=task,model=variant,scope=scope,**score))
    return records


def transfer_rows(variant):
    records=[]
    for path in (OUT/variant/'transfer').glob('*/gat_gine/metrics.json'):
        record=json.loads(path.read_text());task=record['target']
        if task=='GSE154600_EPI_external':
            # A mean probability is the clinical ranking score, not a logit margin.
            for name,scope in [('ensemble_cell_predictions.tsv','external_epithelial_cells'),
                               ('ensemble_patient_predictions.tsv','external_patients')]:
                csv_path=path.with_name(name)
                frame=pd.read_csv(csv_path,sep='\t',float_precision='round_trip')
                if 'margin_resistant_minus_sensitive' in frame:
                    frame=frame.drop(columns='margin_resistant_minus_sensitive')
                    frame.to_csv(csv_path,sep='\t',index=False,float_format='%.17g')
                values=metrics(frame.assign(margin_resistant_minus_sensitive=frame.probability_resistant))
                values['ranking_score']='mean_frozen_source_fold_probability_resistant'
                values['unit']='patient' if scope=='external_patients' else 'cell'
                if scope=='external_patients':
                    values['n_evaluation_units']=values.pop('n_cells')
                    values['n_sensitive_patients']=values.pop('n_sensitive')
                    values['n_resistant_patients']=values.pop('n_resistant')
                record['scopes'][scope]=values
            write_json(path,record)
        for scope,value in record['scopes'].items():
            active=None
            if task=='GSE140440' and scope.startswith('cell_line_'): active='frozen_'+scope
            elif task=='GSE163836' and scope=='primary_FCIBC02_pair': active='frozen_transfer_all_labeled_or_primary_pair'
            elif task=='GSE223779_TS485T' and scope in ['same_drug_crizotinib','cross_drug_alectinib']: active='frozen_TS485T_'+scope
            elif task=='GSE154600_EPI_external' and scope=='external_patients': active='frozen_ovarian_external_patients'
            if active:
                values={k:v for k,v in value.items() if not isinstance(v,(dict,list))}
                if task=='GSE154600_EPI_external':
                    if 'n_cells' in values:
                        values['n_evaluation_units']=values.pop('n_cells')
                        values['n_sensitive_patients']=values.pop('n_sensitive')
                        values['n_resistant_patients']=values.pop('n_resistant')
                    values['unit']='patient'
                    values['ranking_score']='mean_frozen_source_fold_probability_resistant'
                records.append(dict(dataset=task,model=variant,scope=active,**values))
    return records


def report(allow_partial=False,selected_variants=None):
    selected_variants=selected_variants or DEFAULT_VARIANTS
    folder=FIG/'data/gene_embedding';folder.mkdir(parents=True,exist_ok=True)
    baseline=pd.read_csv(FIG/'data/reassessed/primary_comparison.tsv',sep='\t')
    rows=[];status=[];within=[]
    for variant in VARIANTS:
        for job in jobs():
            path=destination(job,variant)/'within'/job['dataset']/'gat_gine/metrics.json'
            complete=path.exists() and 'checkpoint_reload_max_margin_error' in json.loads(path.read_text())
            partial=path.with_name('checkpoint.pt').exists() or path.with_name('training_history.tsv').exists()
            status.append(dict(variant=variant,dataset=job['dataset'],fold=job['fold'],
                included_in_current_phase=variant in selected_variants,complete=complete,
                status='completed' if complete else 'partial_not_completed' if partial else 'not_started'))
    for variant in selected_variants:
        done=True
        for job in jobs():
            path=destination(job,variant)/'within'/job['dataset']/'gat_gine/metrics.json'
            complete=path.exists() and 'checkpoint_reload_max_margin_error' in json.loads(path.read_text())
            if not complete: done=False;continue
            value=json.loads(path.read_text())
            prediction=pd.read_csv(path.with_name('predictions.tsv'),sep='\t',float_precision='round_trip')
            if not prediction.cell_id.is_unique: raise ValueError('Duplicate prediction cell')
            design=pd.read_csv(Path(job['cohort_root'])/'data/cohorts'/job['dataset']/'cells.tsv',sep='\t')
            if prediction.cell_id.tolist()!=design.cell_id.tolist() or prediction.split.tolist()!=design.split.tolist():
                raise ValueError('Saved predictions do not use the original split')
            history=pd.read_csv(path.with_name('training_history.tsv'),sep='\t')
            best=float('inf');selected=None
            for epoch,loss in zip(history.epoch,history.validation_loss):
                if loss<best-.0001: best=loss;selected=int(epoch)
            if value['best_epoch']!=selected: raise ValueError('Checkpoint selection differs')
            for split in ['train','validation','test']:
                part=prediction[prediction.split.eq(split)]
                recomputed=scalar_metrics(part)
                for metric in METRICS:
                    if not np.isclose(recomputed[metric],value[split][metric],rtol=1e-10,atol=1e-10):
                        raise ValueError('Saved split metric differs: '+metric)
                within.append(dict(dataset=job['dataset'],fold=job['fold'],model=variant,split=split,
                                   trainable_parameters=value['trainable_parameter_count'],
                                   n_unique_logit_margins=int(part.margin_resistant_minus_sensitive.nunique()),
                                   logit_margin_std=float(part.margin_resistant_minus_sensitive.std(ddof=0)),
                                   fraction_predicted_positive=float(part.probability_resistant.ge(.5).mean()),
                                   **{k:v for k,v in value[split].items() if not isinstance(v,(dict,list))}))
        if done or allow_partial:
            rows.extend(grouped_rows(variant,allow_partial=allow_partial));rows.extend(transfer_rows(variant))
    pd.DataFrame(status).to_csv(folder/'training_completion.tsv',sep='\t',index=False)
    pd.DataFrame(within).to_csv(folder/'training_split_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    simple=[dict(row,scope='within_cell_split_test') for row in within
            if row['dataset'] in ['GSE140440','GSE223779'] and row['split']=='test']
    for dataset in ['GSE140440','GSE223779']:
        for relation in ['state_15b','esm2_650m','qwen']:
            base=BENCH/'output' if relation=='state_15b' else BENCH/'output/transfer'/relation
            path=base/'within'/dataset/'gat_gine/metrics.json'
            if not path.exists(): continue
            value=json.loads(path.read_text())
            simple.append(dict(dataset=dataset,model=relation,scope='within_cell_split_test',**{
                key:value for key,value in value['test'].items() if not isinstance(value,(dict,list))}))
    pd.DataFrame(simple).to_csv(folder/'within_cell_split_test_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    table=pd.DataFrame(rows)
    design_columns=['endpoint','positive_class','n_sample_sets','n_samples','evaluation_unit',
                    'n_patients','aggregation','n_replicates','cell_line','threshold','resistant_prevalence']
    for index,row in table.iterrows():
        relation,_=VARIANTS[row['model']]
        reference=baseline[(baseline.dataset==row['dataset'])&(baseline.scope==row['scope'])&(baseline.model==relation)]
        if len(reference)!=1: raise ValueError('Missing design-matched reference')
        reference=reference.iloc[0]
        for column in design_columns:
            if column in reference and pd.notna(reference[column]) and (column not in table or pd.isna(table.at[index,column])):
                table.at[index,column]=reference[column]
    phase_status=[s for s in status if s['included_in_current_phase']]
    expected_rows=11*len(selected_variants)
    if not allow_partial and (not all(s['complete'] for s in phase_status) or len(table)!=expected_rows):
        raise ValueError(f'Incomplete phase: {sum(s["complete"] for s in phase_status)}/{len(phase_status)} verified trainings, {len(table)}/{expected_rows} primary results')
    table.to_csv(folder/'pretrained_primary_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    combined=pd.concat([baseline,table],ignore_index=True)
    combined.to_csv(folder/'combined_primary_metrics.tsv',sep='\t',index=False,float_format='%.17g')
    contrasts=[]
    for row in table.to_dict('records'):
        relation,embedding=VARIANTS[row['model']]
        old=baseline[(baseline.dataset==row['dataset'])&(baseline.scope==row['scope'])&(baseline.model==relation)]
        if len(old)!=1: raise ValueError('Missing matched baseline')
        reference=old.iloc[0]
        for count_col in ['n_cells','n_evaluation_units']:
            x,y=row.get(count_col),reference.get(count_col)
            if x is not None and pd.notna(x) and pd.notna(y) and int(x)!=int(y): raise ValueError('Evaluation counts differ')
        contrast=dict(dataset=row['dataset'],scope=row['scope'],model=row['model'],baseline=relation,embedding=embedding)
        for metric in METRICS:
            contrast[metric]=row.get(metric)
            contrast['baseline_'+metric]=reference.get(metric)
            contrast['delta_'+metric]=row.get(metric,np.nan)-reference.get(metric,np.nan)
        contrasts.append(contrast)
    pd.DataFrame(contrasts).to_csv(folder/'matched_feature_differences.tsv',sep='\t',index=False,float_format='%.17g')
    if not allow_partial:
        describe_results(folder,table,pd.DataFrame(contrasts),selected_variants)
    else:
        lines=['# 预训练gene embedding结果解读','',
            '当前阶段仅运行STATE图＋ESM2-15B；gene向量冻结，表达和64维GATv2/GINE主体保留。',
            '',f'当前阶段已核验 {sum(s["complete"] for s in phase_status)}/{len(phase_status)} 项训练，正式评价结果 {len(table)}/{expected_rows} 行。未完成任务不填零，不进入完整折平均。',
            '',*deferred_notes(status),
            '',f'统计与命令见[实验说明]({(BENCH/"code/gene_embedding_exp.md").as_posix()})。最终图表待本阶段结果齐全后生成。']
        (FIG/'gene_embedding结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('VERIFIED_PHASE_TRAININGS',sum(s['complete'] for s in phase_status),'/',len(phase_status),'PRIMARY_ROWS',len(table),flush=True)


def deferred_notes(status):
    lines=['## 其他模型暂缓','',
        'ESM2-650M、Qwen的gene embedding已准备；本阶段不继续训练这两个分类模型。已有完成结果保留，不纳入STATE阶段比较；部分训练不算完成，未开展不填零。',
        '', '|对应图／gene特征|已完成|部分训练、未完成|未开展|','|---|---:|---:|---:|']
    for variant,title in [('esm2_graph__esm2_650m','ESM2-650M'),('qwen_graph__qwen','Qwen')]:
        part=[s for s in status if s['variant']==variant and not s['included_in_current_phase']]
        if not part: continue
        counts=[sum(s['status']==name for s in part) for name in ['completed','partial_not_completed','not_started']]
        lines.append('|'+title+'|'+'|'.join(map(str,counts))+'|')
    deferred=[s for s in status if not s['included_in_current_phase']]
    lines.extend(['','|数据集／任务|ESM2-650M|Qwen|','|---|---|---|'])
    for dataset in sorted({s['dataset'] for s in deferred}):
        cells=[]
        for variant in ['esm2_graph__esm2_650m','qwen_graph__qwen']:
            part=[s for s in deferred if s['dataset']==dataset and s['variant']==variant]
            counts=[sum(s['status']==name for s in part) for name in ['completed','partial_not_completed','not_started']]
            cells.append('；'.join(f'{label}{count}' for label,count in zip(['完成','部分训练','未开展'],counts) if count) or '—')
        lines.append('|'+dataset+'|'+'|'.join(cells)+'|')
    lines.extend(['',f'逐数据集／折状态见[training_completion.tsv]({(FIG/"data/gene_embedding/training_completion.tsv").as_posix()})。部分任务只保留checkpoint和历史，停止后没有优化器现场，未来不宣称精确续训。'])
    return lines


def describe_results(folder,table,changes,selected_variants):
    feature_labels={'state_15b':'ESM2-15B','esm2_650m':'ESM2-650M','qwen':'Qwen'}
    findings=['不同配置按同图基线逐范围比较，不跨终点平均。']
    if selected_variants==['state_graph__state_15b']:
        indexed=changes.set_index('dataset')
        up=int(changes.delta_auroc.gt(1e-12).sum());down=int(changes.delta_auroc.lt(-1e-12).sum())
        ties=len(changes)-up-down
        findings=[f'当前冻结15B节点特征没有显示稳定提升。{len(changes)}个评价范围中，AUROC上升{up}项、下降{down}项、持平{ties}项；这是范围计数，不是独立重复或显著性检验。']
        source=indexed.loc['GSE165897_pre_EOC'];external=indexed.loc['GSE154600_EPI_external']
        n_source=int(table.loc[table.dataset.eq('GSE165897_pre_EOC'),'n_evaluation_units'].iloc[0])
        n_external=int(table.loc[table.dataset.eq('GSE154600_EPI_external'),'n_evaluation_units'].iloc[0])
        findings.extend(['',f'临床患者OOF AUROC：{source.baseline_auroc:.3f}→{source.auroc:.3f}，共{n_source}位患者；独立外部{n_external}位患者：{external.baseline_auroc:.3f}→{external.auroc:.3f}。个别相对上升需结合绝对性能、小样本及外部结果，不足以认定泛化优势。'])
    lines=['# 预训练gene embedding对照结果','',
        '本轮比较冻结的蛋白衍生gene向量与原16维可训练gene identity。表达值、signature节点、K、划分、训练预算及GATv2/GINE主体保持一致；高维向量直接与表达拼接投影到64维。',
        '', f'本阶段{14*len(selected_variants)}个训练任务已完成checkpoint重载核验；11个评价范围×{len(selected_variants)}个配置，与旧结果共同保存在 `data/gene_embedding/combined_primary_metrics.tsv`。',
        '', '## 当前结果支持什么','',
        *findings,
        '', '## 对应图的同范围比较','',
        '|数据集／范围|'+'|'.join(feature_labels[VARIANTS[v][1]]+' AUROC（变化）' for v in selected_variants)+'|',
        '|---|'+'---:|'*len(selected_variants)]
    for (dataset,scope),part in changes.groupby(['dataset','scope'],sort=True):
        indexed=part.set_index('model');cells=[]
        for variant in selected_variants:
            row=indexed.loc[variant];cells.append(f'{row.auroc:.3f} ({row.delta_auroc:+.3f})')
        lines.append('|'+dataset+' / '+scope+'|'+'|'.join(cells)+'|')
    sections=[
        ('18_pretrained_gene_features_matched_topologies.pdf','对应图的配对比较',
         '当前阶段比较对应图的原16维向量和冻结预训练向量，节点、边及测试范围不变。前5行为分组within，后6行为冻结迁移，与差值图行序一致。AUROC使用阳性减阴性logit，BA阈值0.5，AP结合阳性基率。临床按患者平均细胞概率评价，PC9两个样本集合、ReSis两个重复分别等权平均；其余迁移按指定范围全部有标签细胞，不跨终点平均。'),
        ('19_pretrained_gene_features_matched_differences.pdf','同图节点特征增益',
         '每格为同一数据、scope和图的预训练向量模型减原16维向量模型。统计单位与18图一致；AUROC、BA分别计算差值。单种子不作显著性检验，接近随机时的正差值不视为有效泛化。色阶±0.4，标注实际值。'),
        ('20_pretrained_gene_features_within_cell_split.pdf','旧队列内测试',
         'GSE140440和FA34沿用原随机细胞测试划分，train/validation/test完整指标另见training_split_metrics.tsv。GSE140440缺少旧ESM/Qwen同图基线，显示NA，不计算纯embedding增益。随机细胞划分不能证明跨样本或患者泛化。')]
    for name,title,text in sections:
        path=(FIG/'figures'/name).as_posix()
        lines.extend(['',f'## [{title}]({path})','',text,
                      '', '合法零值保留；不可定义指标为缺失，图中显示NA，不用0填充。'])
    lines.extend(['','## 结论边界','',
        '向量维数和冻结方式不同，投影层参数量也不同；因此收益不能单独证明蛋白生物语义是唯一原因。当前没有同维随机／打乱gene映射对照，不能区分语义信息与一般特征容量。单种子结果不用于宣称稳定最优。',
        '', '全维gene向量直接拼接也改变了投影层初始化尺度及表达通道的相对影响；本结果适用于当前编码和训练设置，不能将退化直接解释为15B表示没有生物学信息。',
        '', '三划分统计另记录logit差的唯一值数、标准差及阳性预测比例，用于识别常量输出或单类预测。常量输出的AUROC为0.5，反映该配置在当前训练设置下未形成有效区分，不据此断言其预训练表示没有生物学信息。',
        '', 'PC9为DTEP治疗状态，NK为杀伤存活；FCIBC02存在批次混杂，TS485T只有一个外部类器官来源，临床外部仅4个有标签患者。GSE202234仅作身份／处理诊断，PANC-1不恢复到正式主比较。',
        '', f'复现：[实验命令]({(BENCH/"code/gene_embedding_exp.md").as_posix()})；统计scNET，绘图single_cell，无新增依赖。',
        '', f'统计表：[同范围差值]({(folder/"matched_feature_differences.tsv").as_posix()})；[三划分指标]({(folder/"training_split_metrics.tsv").as_posix()})。'])
    lines.extend(['',f'核查：[14项输入与checkpoint审计]({(folder/"completed_phase_validation.tsv").as_posix()})；[旧统计回算]({(folder/"reference_aggregation_audit.tsv").as_posix()})。'])
    (FIG/'gene_embedding结果解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    completion=pd.read_csv(folder/'training_completion.tsv',sep='\t').to_dict('records')
    with (FIG/'gene_embedding结果解读.md').open('a',encoding='utf-8') as stream:
        stream.write('\n'+'\n'.join(deferred_notes(completion))+'\n')
    integrated=FIG/'全部结果整合.md'
    heading='## 预训练gene节点特征对照'
    original=integrated.read_text(encoding='utf-8').split('\n'+heading)[0].rstrip()
    summary=['',heading,'',
        f'本阶段新增{14*len(selected_variants)}个训练任务和冻结迁移，保留表达、signature、K、划分与64维GATv2/GINE主体。当前只开展STATE图＋ESM2-15B，其他对应模型暂缓。',
        '', '|对应图的gene特征|AUROC高于同图旧16维向量的范围数|', '|---|---:|']
    for model,title in [('esm2_graph__esm2_650m','ESM2-650M'),('qwen_graph__qwen','Qwen'),('state_graph__state_15b','ESM2-15B')]:
        if model not in selected_variants: continue
        part=changes[changes.model.eq(model)]
        summary.append(f'|{title}|{int(part.delta_auroc.gt(0).sum())}/{len(part)}|')
    summary.extend(['',
        '以上为范围计数，非独立重复次数或显著性检验；TS485T两个范围共享未处理细胞。维数、冻结方式与参数量不同，不能单独归因于蛋白语义；最终解读结合BA、AP、样本背景与外部验证。',
        '', f'逐范围结果、18–20图及统计方法见[gene embedding结果解读]({(FIG/"gene_embedding结果解读.md").as_posix()})。'])
    integrated.write_text(original+'\n'+'\n'.join(summary)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--allow_partial',action='store_true')
    parser.add_argument('--variant',nargs='+',choices=list(VARIANTS),default=DEFAULT_VARIANTS)
    args=parser.parse_args();report(args.allow_partial,args.variant)
