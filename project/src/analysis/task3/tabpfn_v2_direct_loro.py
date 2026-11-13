"""任务3 第20步：直接使用Nature 2025 TabPFN-v2官方模型做固定LORO评估。

只调用作者发布的分类器和2025年原始v2权重，不修改网络或添加项目损失。
不重复第7步的线性/树模型，不重训完整V5；旧V5站点AUC直接读取第17步汇总。
TabPFN是小中型表格预测模型，并非专为低维因果解耦设计；本步只验证预测适用性。
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'task2'))
import east_pair_distance_error_decomposition as prior

audit=prior.audit
HF_REVISION='275f19d2add9c2c0de38f9ebaded3856599fa2e7'
CHECKPOINT_SHA='f65a35685aeef42e31b796d9bfa34e68d6fc780bc98e7bff7763802964cf435f'
CHECKPOINT_URL=f'https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/{HF_REVISION}/tabpfn-v2-classifier.ckpt'


def check_runtime():
    """任务3 第20步：锁定2025年实现，禁止新版默认模型被误称为v2。"""
    try:
        versions={n:importlib.metadata.version(n) for n in ('tabpfn','scikit-learn','torch','numpy','scipy','pandas')}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError('缺少TabPFN独立环境依赖，请先安装配套tabpfn_v2_requirements.txt。') from exc
    if versions['tabpfn']!='2.0.9' or versions['scikit-learn']!='1.6.1':
        raise RuntimeError('本实验需要独立环境tabpfn==2.0.9、scikit-learn==1.6.1；请按配套依赖安装，不修改原研究环境。')
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier,versions


def checkpoint(path):
    """任务3 第20步：从固定作者仓库下载原始v2，并校验公开LFS内容哈希。"""
    path=path.expanduser().resolve()
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_suffix('.download')
        print('任务3 第20步：下载作者原始TabPFN-v2权重，约29MB；不上传研究数据',flush=True)
        try:
            with urllib.request.urlopen(CHECKPOINT_URL,timeout=60) as response,partial.open('wb') as stream:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    stream.write(block)
            if audit.digest(partial)!=CHECKPOINT_SHA:raise ValueError('权重内容校验不通过')
            partial.replace(path)
        except Exception as exc:
            if partial.exists():partial.unlink()
            raise RuntimeError('官方权重下载失败；可从固定URL下载后使用--checkpoint指定本地文件：'+CHECKPOINT_URL) from exc
    if audit.digest(path)!=CHECKPOINT_SHA:raise ValueError('权重不是本实验固定的2025原始v2文件；不会自动改用其他模型')
    return path


def new_model(cls,path,device):
    return cls(model_path=path,device=device,n_estimators=4,random_state=42,
               categorical_features_indices=None,ignore_pretraining_limits=True,
               n_jobs=2,fit_mode='fit_preprocessors')


def predict(model,x,batch_size):
    """任务3 第20步：全体样本均评估，分批仅限制查询内存，不抽样训练人群。"""
    column=list(model.classes_).index(1);values=[]
    for start in range(0,len(x),batch_size):
        p=np.asarray(model.predict_proba(x[start:start+batch_size]))[:,column]
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('官方模型返回无效概率')
        values.append(p)
        print(f'任务3 第20步：已预测{min(start+batch_size,len(x))}/{len(x)}人',flush=True)
    return np.concatenate(values)


def metric(y,p):
    both=len(np.unique(y))==2
    return dict(auc=float(roc_auc_score(y,p)) if both else np.nan,
                auprc=float(average_precision_score(y,p)) if both else np.nan,
                brier=float(brier_score_loss(y,p)),n=len(y),positive_n=int(y.sum()))


def rows(region,stage,y,sites,p):
    result=[dict(held_out=region,stage=stage,scope='region_pooled',group='REGION_ALL',**metric(y,p))]
    per_site=[]
    for site in sorted(set(sites)):
        use=sites==site;per_site.append(dict(held_out=region,stage=stage,scope='site',group=str(site),**metric(y[use],p[use])))
    result.extend(per_site)
    result.append(dict(held_out=region,stage=stage,scope='equal_site_macro',group='SITE_MACRO',
        **{k:float(np.mean([r[k] for r in per_site])) if all(np.isfinite(r[k]) for r in per_site) else np.nan for k in ('auc','auprc','brier')}))
    return result


def comparison(metrics_table,old_bins):
    base=old_bins[(old_bins.stage=='test')&(old_bins.distance=='phone_balanced')].groupby(['held_out','site'],as_index=False).error_contribution.sum()
    base['v5_auc']=1-base.error_contribution;base=base.rename(columns={'site':'group'})
    q=metrics_table[(metrics_table.stage=='test')&(metrics_table.scope=='site')].merge(base[['held_out','group','v5_auc']],on=['held_out','group'],validate='one_to_one')
    if len(q)!=12:raise ValueError('留出评估必须完整覆盖12个站点')
    q['delta_auc']=q.auc-q.v5_auc
    return q


def draw(table,effect,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4));q=effect.copy();q['name']=q.held_out+':'+q.group;q=q.sort_values('name')
    ax.bar(q.name,q.delta_auc);ax.axhline(0,color='grey');ax.tick_params(axis='x',rotation=50)
    ax.set_ylabel('AUC change vs saved full V5');ax.set_title('Official TabPFN-v2; fixed model, no target selection')
    fig.tight_layout();fig.savefig(out/'figures/01_site_auc_vs_existing_v5.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    q=effect.groupby('held_out')[['auc','v5_auc']].mean().reindex(audit.REGIONS)
    q.plot.bar(ax=axes[0],rot=0);axes[0].set_ylabel('Equal-site macro AUC')
    t=table[(table.stage=='test')&(table.scope=='equal_site_macro')].set_index('held_out').reindex(audit.REGIONS)
    t[['auprc','brier']].plot.bar(ax=axes[1],rot=0);axes[1].set_title('AUPRC higher; Brier lower')
    fig.tight_layout();fig.savefig(out/'figures/02_region_ranking_and_probability.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=audit.ROOT/'project/data/data_mission2.csv')
    parser.add_argument('--previous',type=Path,default=audit.ROOT/'outputs/v2-region-loro-selection91-42')
    parser.add_argument('--distance-step',type=Path,default=audit.ROOT/'outputs/analysis_2611/task2_step17_east_pair_distance_error_decomposition')
    parser.add_argument('--output',type=Path,default=audit.ROOT/'outputs/analysis_2611/task3_step20_tabpfn_v2_direct_loro')
    parser.add_argument('--checkpoint',type=Path,default=Path.home()/'.cache/elderly_health/tabpfn_v2/tabpfn-v2-classifier.ckpt')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--query-batch-size',type=int,default=128)
    parser.add_argument('--smoke-test',action='store_true',help='仅用合成数据验证官方权重和真实推理，不读研究数据')
    args=parser.parse_args()
    if args.query_batch_size<1:raise ValueError('查询批量必须为正整数')
    cls,versions=check_runtime();weight=checkpoint(args.checkpoint)
    torch.set_num_threads(2);os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET']='1'
    if args.smoke_test:
        rng=np.random.default_rng(42);x=rng.normal(size=(64,19));y=(x[:,0]+x[:,1]>0).astype(int)
        m=new_model(cls,weight,args.device);m.fit(x[:48],y[:48]);p=predict(m,x[48:],args.query_batch_size)
        if p.shape!=(16,):raise RuntimeError('官方模型推理形状异常')
        print('任务3 第20步：官方2.0.9代码和原始v2权重真实推理通过；尚未产生健康数据实验结果',flush=True)
        return
    if audit.digest(args.data)!=audit.EXPECTED_HASH:raise ValueError('研究数据版本不一致')
    rules=json.loads((args.distance_step/'source_distance_rules.json').read_text(encoding='utf-8-sig'))
    old_meta=json.loads((args.distance_step/'experiment_metadata.json').read_text(encoding='utf-8-sig'))
    if not old_meta.get('completed') or old_meta['input_sha256']!=audit.EXPECTED_HASH:raise ValueError('已有完整V5参照数据版本无效')
    for region in audit.REGIONS:
        for name in ('config.json','splits.npz'):
            if audit.digest(args.previous/region/'v5'/name)!=rules[region]['source_artifacts'][name]:raise ValueError('历史划分版本不一致')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'figures').mkdir()
    metadata=dict(completed=False,paper='https://doi.org/10.1038/s41586-024-08328-6',paper_year=2025,
        method='official_TabPFN_v2',package_versions=versions,checkpoint_sha256=CHECKPOINT_SHA,checkpoint_url=CHECKPOINT_URL,
        input_sha256=audit.EXPECTED_HASH,runner_sha256=audit.digest(Path(__file__)),n_estimators=4,seed=42,
        device=args.device,query_batch_size=args.query_batch_size,pretrained_parameter_updates=0,v5_training_runs=0,
        context_fits=0,individual_records_exported=False,source_validation_used_for_selection=False,
        limitations=[
            '小中型表格预测方法，适用规模覆盖19维；不是专门为低维因果解耦设计，也不识别因果特征。',
            '直接使用作者分类器、内部预处理和2025原始v2权重；未添加加权BCE、IRM、稀疏或本项目校正头。',
            '输入为原19项原始数值和缺失，采用官方自动类别检测；患者划分与完整V5相同，预处理及训练目标不同。',
            '固定默认四成员集成、固定种子、无超参搜索；这是作者方法的有限配置评估，不是Nature整套基准复现。',
            '只用来源训练样本作为上下文；未将来源验证并入，不使用目标标签训练、校准或选模。',
            '不重训线性、树或V5；V5站点AUC由旧汇总读取，差异不能单独归因于表征崩溃。',
            'CPU使用完整来源训练人群，可能耗时较长；查询分批仅控制内存，不减少样本。',
            '既有数据已反复探索，单固定配置结果只作方法适用性线索。'])
    audit.dump(args.output/'experiment_metadata.json',metadata)
    frame=prior.read_raw(args.data);cache={};results=[];runtime=[]
    for region in audit.REGIONS:
        folder=args.previous/region/'v5'
        train,_,y,sites=prior.load_part(folder,'train',frame)
        val,_,vy,vs=prior.load_part(folder,'val',frame)
        with np.load(folder/'splits.npz',allow_pickle=False) as z:
            if any(set(z['sample_id_train'])&set(z['sample_id_'+s]) for s in ('val','test')):raise ValueError('来源划分与其他人群重叠')
        x=train[list(audit.FEATURES)].to_numpy(float);vx=val[list(audit.FEATURES)].to_numpy(float)
        start=time.perf_counter();model=new_model(cls,weight,args.device)
        print(f'任务3 第20步：留出{region}，以{len(x)}名来源训练者作为官方模型上下文，不做参数更新',flush=True)
        model.fit(x,y);fit_seconds=time.perf_counter()-start
        pv=predict(model,vx,args.query_batch_size);results.extend(rows(region,'source_validation',vy,vs,pv))
        runtime.append(dict(held_out=region,source_n=len(x),validation_n=len(vx),context_fit_seconds=fit_seconds,
            inferred_categorical_features='|'.join(audit.FEATURES[i] for i in model.inferred_categorical_indices_)))
        cache[region]=model;metadata['context_fits']+=1
        pd.DataFrame(results).to_csv(args.output/'metrics.csv',index=False,encoding='utf-8-sig')
    audit.dump(args.output/'source_phase_completed.json',dict(completed=True,hyperparameter_search=False))
    for region in audit.REGIONS:
        target,_,y,sites=prior.load_part(args.previous/region/'v5','test',frame)
        start=time.perf_counter();p=predict(cache[region],target[list(audit.FEATURES)].to_numpy(float),args.query_batch_size)
        results.extend(rows(region,'test',y,sites,p));runtime[audit.REGIONS.index(region)]['target_predict_seconds']=time.perf_counter()-start
    table=pd.DataFrame(results);effect=comparison(table,pd.read_csv(args.distance_step/'distance_pair_bins.csv',dtype={'site':str}))
    table.to_csv(args.output/'metrics.csv',index=False,encoding='utf-8-sig')
    effect.to_csv(args.output/'site_auc_vs_existing_v5.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(runtime).to_csv(args.output/'runtime_and_encoding.csv',index=False,encoding='utf-8-sig')
    draw(table,effect,args.output);metadata['completed']=True;audit.dump(args.output/'experiment_metadata.json',metadata)
    print(f'任务3 第20步完成：官方v2三折评估，预训练参数更新0次，已输出2张图。目录：{args.output}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
